"""`GET /worlds`: the index a viewer opens old worlds from."""

import json
import pathlib
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.result_channel_fixtures import build_world
from tower.results.world_builder_library import WORLDS_CONTRACT, build_world_listing
from tower.routes import geometry as geometry_routes
from tower.world_builder.records import Session, World
from tower.world_builder.store import WorldStore


def _client(store) -> TestClient:
    app = FastAPI()
    app.include_router(geometry_routes.router)
    app.state.world_root = store.root
    return TestClient(app)


def test_listing_names_the_derived_world_and_its_session(derived_world):
    store, world_id, session_id = derived_world
    listing = build_world_listing(store)
    assert listing["contract"] == WORLDS_CONTRACT
    assert listing["world_count"] == 1
    world = listing["worlds"][0]
    assert world["world_id"] == world_id
    assert world["live"] is False
    assert world["session_count"] == 1
    session = world["sessions"][0]
    assert session["session_id"] == session_id
    assert session["has_geometry"] is True
    assert "ended_at" in session and "end_reason" in session


def test_a_live_lock_marks_the_world_live_and_a_dead_one_does_not(derived_world):
    store, world_id, _ = derived_world
    lock = store.lock_path(world_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    # A pid that cannot be running: pid 0 is never a user process holder.
    lock.write_text(json.dumps({"pid": 999999999}))
    assert build_world_listing(store)["worlds"][0]["live"] is False
    # This process is running, but it is *us*; a viewer asking from inside
    # the writer would not call that "live", and the store's own lock
    # logic makes the same exception.
    lock.write_text(json.dumps({"pid": os.getpid()}))
    assert build_world_listing(store)["worlds"][0]["live"] is False


def test_an_open_session_whose_writer_is_dead_is_abandoned(derived_world):
    """A builder killed mid-walk (Tower shutdown grace, a hard kill) never
    reaches `stop_session`, so its record keeps `ended_at: null` forever.
    The listing must not let that read as "still open"."""
    from tower.world_builder.store import Session

    store, world_id, session_id = derived_world
    # Finished session, no lock: not abandoned.
    assert build_world_listing(store)["worlds"][0]["sessions"][0]["abandoned"] is False

    store.write_session(Session(session_id=session_id, world_id=world_id,
                                started_at=1.0, ended_at=None))
    lock = store.lock_path(world_id)
    lock.parent.mkdir(parents=True, exist_ok=True)
    # Open record, lock held by a pid that is not running: abandoned.
    lock.write_text(json.dumps({"pid": 999999999}))
    world = build_world_listing(store)["worlds"][0]
    assert world["live"] is False
    assert world["sessions"][0]["ended_at"] is None
    assert world["sessions"][0]["abandoned"] is True
    # Open record, no lock at all (an older layout, or a reclaimed lock):
    # nobody is writing it, so it is abandoned too.
    lock.unlink()
    assert build_world_listing(store)["worlds"][0]["sessions"][0]["abandoned"] is True
    # Open record, lock held by a running pid that is not us: live, not
    # abandoned. The parent of this test process is running by definition.
    lock.write_text(json.dumps({"pid": os.getppid()}))
    world = build_world_listing(store)["worlds"][0]
    assert world["live"] is True
    assert world["sessions"][0]["abandoned"] is False


def test_route_serves_the_listing_and_404s_without_a_root(derived_world):
    store, world_id, _ = derived_world
    client = _client(store)
    response = client.get("/worlds")
    assert response.status_code == 200
    body = response.json()
    assert body["contract"] == WORLDS_CONTRACT
    assert body["worlds"][0]["world_id"] == world_id

    app = FastAPI()
    app.include_router(geometry_routes.router)
    assert TestClient(app).get("/worlds").status_code == 404


def test_an_unreadable_world_is_omitted_not_invented(derived_world, tmp_path):
    store, world_id, _ = derived_world
    broken = store.world_dir("deadbeefdeadbeefdeadbeefdeadbeef")
    broken.mkdir(parents=True)
    (broken / "world.json").write_text("{not json")
    listing = build_world_listing(store)
    assert [w["world_id"] for w in listing["worlds"]] == [world_id]


def test_worlds_are_newest_first_and_sessions_oldest_first(tmp_path):
    """RECENCY, established rather than assumed.

    This created two worlds back-to-back and asserted the second sorted
    first. Windows' clock granularity is about 15.6 ms, so under load the
    two shared an `updated_at` -- and "newest first" is not a property the
    system can have when two things are the same age. It failed in the
    suite as `assert 1 < 0` and it was the TEST that was wrong: the
    listing was answering an unanswerable question.

    `test_two_worlds_in_one_clock_tick_keep_a_stable_order` covers what is
    actually guaranteed for a tie -- an order that does not change between
    polls, so rows do not swap under the wearer's finger.
    """
    from dataclasses import replace

    from tower.world_builder.engine import WorldBuilderEngine

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    first = engine.create_world("first")
    second = engine.create_world("second")
    for world_id, stamp in ((first, 1000.0), (second, 2000.0)):
        world = store.read_world(world_id)
        store.write_world(replace(world, created_at=stamp, updated_at=stamp))
    listing = build_world_listing(store)
    ids = [w["world_id"] for w in listing["worlds"]]
    assert ids.index(second) < ids.index(first)
    assert listing["worlds"][0]["display_name"] == "second"
    assert all(w["sessions"] == [] for w in listing["worlds"])


def test_two_worlds_in_one_clock_tick_keep_a_stable_order(tmp_path):
    """A tie must not leave the order to whatever the filesystem yields.

    Windows' clock granularity is about 15.6 ms, so two worlds created
    back-to-back can share an `updated_at`. Sorting on that alone left
    their order to `list_world_ids`, which can differ between polls -- and
    the phone redraws this list every time it arrives, so rows swap places
    under the wearer's finger. It surfaced first as a suite flake under
    load (`assert 1 < 0` on exactly this pair).
    """
    from dataclasses import replace

    from tower.world_builder.engine import WorldBuilderEngine

    store = WorldStore(tmp_path)
    engine = WorldBuilderEngine(store)
    ids = [engine.create_world(f"world-{index}") for index in range(4)]

    # Force the tie the clock only sometimes produces.
    for world_id in ids:
        world = store.read_world(world_id)
        store.write_world(replace(world, updated_at=1000.0, created_at=1000.0))

    orders = {
        tuple(w["world_id"] for w in build_world_listing(store)["worlds"])
        for _ in range(5)
    }
    assert len(orders) == 1, f"the listing order was not stable: {orders}"
    # And it is a real order, not insertion luck: reversed ids, same answer.
    assert sorted(next(iter(orders)), reverse=True) == list(next(iter(orders)))


# -- round 16: the picker's own two defects ----------------------------


def test_a_session_whose_build_output_is_gone_is_not_called_unbuilt(derived_world):
    """`unbuilt` claims nothing ever ran. A manifest proves otherwise.

    `session_state`'s docstring says it "mirrors `_lifecycle`". It did
    not: `_lifecycle` grew a whole `interrupted` branch precisely so a
    session whose manifest describes a build would stop being reported as
    one that never built -- and this surface, the one a person chooses a
    walk FROM, kept the old sentence. Found by a reviewer building the
    state and reading both answers side by side.
    """
    store, world_id, session_id = derived_world
    derived = store.derived_dir(world_id) / session_id

    # A build ran -- its manifest is the proof, and it is still there --
    # and its output is gone.
    (derived / "poses.json").unlink()
    (derived / "points.json").unlink()
    assert (derived / "manifest.json").exists()

    session = build_world_listing(store)["worlds"][0]["sessions"][0]
    assert session["has_geometry"] is False
    assert session["state"] == "interrupted", (
        "a session whose manifest proves a build ran was described to the "
        "wearer as one that never built"
    )


def test_a_session_that_truly_never_built_is_still_called_unbuilt(derived_world):
    """The other half. `interrupted` must not swallow `unbuilt` whole."""
    store, world_id, session_id = derived_world
    derived = store.derived_dir(world_id) / session_id
    for name in ("poses.json", "points.json", "manifest.json"):
        (derived / name).unlink()
    (store.derived_dir(world_id) / "manifest.json").unlink()

    session = build_world_listing(store)["worlds"][0]["sessions"][0]
    assert session["has_geometry"] is False
    assert session["state"] == "unbuilt"


def test_one_malformed_world_does_not_take_the_whole_listing_down(derived_world):
    """The tiebreak made a latent bad row fatal, and only on a tie.

    `created_at` is required and un-defaulted, so a world MISSING it never
    reaches the sort -- `world_from_json_dict` raises `KeyError` and
    `build_world_listing` skips it. A world carrying `null` there does
    reach it, and comparing `None` to a float raises `TypeError` only
    when a tie on `updated_at` sends Python to the second key. The sort is
    outside the try/except and `routes/geometry.py` has no handler, so
    that is a 500 on `GET /worlds`: one bad row and the picker loses every
    world.

    The single-key sort this replaced could not reach it, which is what
    makes it a regression rather than an old wart.
    """
    store, world_id, _ = derived_world
    original = json.loads((store.world_dir(world_id) / "world.json").read_text())

    # A second world, TIED on updated_at -- which on Windows' ~15.6 ms
    # clock two worlds created back to back genuinely are.
    other = dict(original)
    other["world_id"] = "w1"
    other["created_at"] = None
    other["session_ids"] = []
    path = store.world_dir("w1") / "world.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(other))

    listing = build_world_listing(store)
    assert listing["world_count"] == 2, (
        "a malformed row took the whole listing with it"
    )


def test_the_recount_agrees_with_a_manifest_written_by_hand(derived_world):
    """`_summarise_pose_rows` against figures nothing in it produced.

    The fixture's manifest is written by a person, from a pose layout
    designed around the one rule that is easy to get wrong: an ANCHOR is
    a position only in a segment that solved. Segment 0 has an anchor and
    a solved pose (2 positions); segment 1 has an anchor and a refusal
    (0). `poses_positioned: 2`, not 4, not 3.

    Recomputing that from poses.json is what lets a world with no
    manifest report real figures, so the recount has to agree with the
    build. This compares it against a number the recount had no hand in.
    """
    from tower.results.world_builder import _summarise_pose_rows

    store, world_id, session_id = derived_world
    manifest = store.read_session_manifest(world_id, session_id)
    recount = _summarise_pose_rows(
        store.derived_dir(world_id) / session_id / "poses.json"
    )
    for field in (
        "poses_solved", "poses_refused", "poses_anchor",
        "poses_positioned", "segments",
    ):
        assert recount[field] == manifest[field], field

def test_a_build_that_found_nothing_says_no_geometry_not_interrupted(derived_world):
    """`engine.build` writes a derived tree even when it solved nothing.

    `points.json` is then `{"points": []}` -- 14 bytes -- and **eleven
    sessions on the real 163-world root are exactly that shape**. They
    listed as `complete` with `has_geometry: true`, because that field
    used to mean "the files exist", so the picker told the wearer opening
    the walk would show something while the panel behind it projected
    `needsRetry`. `WorldPickerView` branches on that field.

    And the first fix for it called them `interrupted`, which iOS renders
    "Interrupted": a claim that the walk FAILED, over one that finalized
    cleanly and merely found nothing to reconstruct. `unbuilt` renders
    "No geometry", which is the true sentence.
    """
    store, world_id, session_id = derived_world
    derived = store.derived_dir(world_id) / session_id

    # A build that ran and found nothing: the tree is there and empty,
    # and the manifest says so rather than being absent.
    (derived / "points.json").write_text(json.dumps({"points": []}))
    (derived / "poses.json").write_text(json.dumps({"poses": []}))
    for path in (derived / "manifest.json", store.derived_dir(world_id) / "manifest.json"):
        manifest = json.loads(path.read_text())
        manifest["points"] = 0
        manifest["poses_solved"] = 0
        manifest["poses_positioned"] = 0
        path.write_text(json.dumps(manifest))

    session = build_world_listing(store)["worlds"][0]["sessions"][0]
    assert session["has_geometry"] is False, (
        "the picker promised the wearer something to look at, over an "
        "empty reconstruction"
    )
    assert session["state"] == "unbuilt", (
        "a walk that finalized cleanly and found nothing was reported as "
        "an interruption"
    )


def test_the_picker_and_the_panel_agree_about_every_session(derived_world):
    """A row that says one thing must not open onto a canvas saying another.

    `session_state`'s own comment: "A row in the picker and the panel it
    opens must not disagree about what a session is." The two surfaces
    compute their answers separately, from the same disk, and the only
    thing keeping them together is that they ask the same question of the
    same numbers. This asserts they do, over the three shapes that exist.
    """
    from tower.results.world_builder import _has_drawable_geometry

    store, world_id, session_id = derived_world
    derived = store.derived_dir(world_id) / session_id

    def row():
        return build_world_listing(store)["worlds"][0]["sessions"][0]

    # 1. A real build.
    assert row()["has_geometry"] is True
    assert _has_drawable_geometry({
        "geometry": {"element_count": 2}, "trajectory": {"pose_count": 2},
    }) is True

    # 2. A build that found nothing.
    (derived / "points.json").write_text(json.dumps({"points": []}))
    (derived / "poses.json").write_text(json.dumps({"poses": []}))
    for path in (derived / "manifest.json", store.derived_dir(world_id) / "manifest.json"):
        manifest = json.loads(path.read_text())
        manifest.update(points=0, poses_solved=0, poses_positioned=0)
        path.write_text(json.dumps(manifest))
    assert row()["has_geometry"] is False
    assert _has_drawable_geometry({
        "geometry": {"element_count": 0}, "trajectory": {"pose_count": 0},
    }) is False

    # 3. Poses but no points, and points but no poses. EITHER is geometry
    #    -- a build can place cameras and recover few points, or recover
    #    points across segments whose cameras were never placed -- and
    #    both surfaces have to agree about that too.
    for points, positioned in ((5, 0), (0, 5)):
        for path in (
            derived / "manifest.json",
            store.derived_dir(world_id) / "manifest.json",
        ):
            manifest = json.loads(path.read_text())
            manifest.update(points=points, poses_positioned=positioned)
            path.write_text(json.dumps(manifest))
        assert row()["has_geometry"] is True, (points, positioned)
        assert _has_drawable_geometry({
            "geometry": {"element_count": points},
            "trajectory": {"pose_count": positioned},
        }) is True, (points, positioned)


def test_a_legacy_world_the_channel_says_to_open_is_actually_served(tmp_path):
    """The wearer's NEXT TAP, which is where four rounds of fixes stopped short.

    The status channel saying `ready` with real figures is a promise, and
    the phone redeems it by fetching from `routes/geometry.py`. Two
    campaign defects lived in exactly that gap: `ready` beside a 404 when
    `read_derived`'s verify gate judged an older session against another
    session's digest, and a route serving geometry the phone had been told
    was still finalizing.

    This walks the whole promise for the shape round 16 added: a derived
    tree with no manifest at all.
    """
    import json

    from tower.results.world_builder import WorldBuilderStatusProducer

    root = tmp_path / "worlds"
    world_id, session_id = build_world(root, frames=8)
    store = WorldStore(root)
    derived = store.derived_dir(world_id)
    (derived / session_id / "manifest.json").unlink()
    (derived / "manifest.json").unlink()

    # 1. The channel says there is something here, and how much.
    producer = WorldBuilderStatusProducer(root, lambda: 1000.0)
    snapshot = producer.snapshot(world_id=world_id, session_id=session_id)
    payload = getattr(snapshot, "payload", snapshot)
    assert payload["lifecycle"]["state"] == "ready"
    promised = payload["geometry"]["element_count"]
    assert promised > 0

    # 2. The route hands it over. Same disk, separate reader, no manifest
    #    for either of them to agree through.
    client = _client(store)
    manifest = client.get(
        f"/worlds/{world_id}/geometry/manifest", params={"session_id": session_id}
    )
    assert manifest.status_code == 200, manifest.text
    body = manifest.json()
    assert body["segment_count"] >= 1

    served = 0
    for index in range(body["segment_count"]):
        segment = client.get(
            f"/worlds/{world_id}/geometry/segment/{index}",
            params={"session_id": session_id},
        )
        assert segment.status_code == 200, segment.text
        served += len(segment.json()["points"])

    assert served == promised, (
        f"the channel promised {promised} points and the route served {served}"
    )
    # And the promise came from the files, not from either of them
    # trusting the other.
    on_disk = len(
        json.loads((derived / session_id / "points.json").read_text())["points"]
    )
    assert on_disk == promised


def test_a_malformed_updated_at_does_not_take_the_top_of_the_picker(derived_world):
    """"Newest first" must not be won by a value that is not a time.

    The first tiebreak keyed on the TYPE NAME -- `("str", ...)` above
    `("num", ...)` under `reverse=True` -- so a world carrying
    `updated_at: "2020-01-01T00:00:00Z"` was placed ahead of every real
    world. A reviewer built it against a copy of the real 163-world root
    and watched it take the top row. A malformed value is not evidence of
    recency.
    """
    store, world_id, _ = derived_world
    original = json.loads((store.world_dir(world_id) / "world.json").read_text())

    for bad in ("2020-01-01T00:00:00Z", None, {"a": 1}, True):
        other = dict(original, world_id="w1", session_ids=[], updated_at=bad)
        path = store.world_dir("w1") / "world.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(other))

        listing = build_world_listing(store)
        assert listing["world_count"] == 2, bad
        assert listing["worlds"][0]["world_id"] == world_id, (
            f"a world whose updated_at is {bad!r} took the top row"
        )


def test_no_value_of_updated_at_can_take_the_listing_down(derived_world):
    """`_sortable`'s docstring says it never raises. It has to be true.

    `float(10**400)` raises `OverflowError` -- out of the sort, out of
    `build_world_listing`, which has no handler above it, and into a 500
    on `GET /worlds` that loses every world. Same trigger as the case the
    tiebreak was written to fix. NaN is here too: it does not raise, it
    silently makes `sort` produce an arbitrary permutation.
    """
    store, world_id, _ = derived_world
    original = json.loads((store.world_dir(world_id) / "world.json").read_text())

    for bad in (10**400, float("nan"), float("inf"), [], "", 0):
        other = dict(
            original, world_id="w1", session_ids=[],
            updated_at=bad if not isinstance(bad, float) or bad == bad else 0,
            created_at=bad,
        )
        # `created_at` carries the hostile value so the tie on
        # `updated_at` is what forces the comparison, which is the only
        # way the second key is ever reached.
        other["updated_at"] = original["updated_at"]
        path = store.world_dir("w1") / "world.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(other))

        listing = build_world_listing(store)
        assert listing["world_count"] == 2, bad


# -- round 18 ----------------------------------------------------------


def test_one_bad_session_record_does_not_empty_saved_worlds(derived_world):
    """The session sort, thirty lines above the world sort that was fixed.

    `session_from_json_dict` does not coerce -- `started_at =
    data["started_at"]`, raw -- so a `session.json` carrying a string
    reaches the comparison. The sort sits at the top of the per-world
    loop, OUTSIDE the inner try that skips an unreadable session, outside
    `build_world_listing`'s only handler, on a route with none. So the
    raise escapes before ANY world is returned: one bad record and the
    picker loses all 163 worlds, not one row.

    A reviewer found it by applying the world sort's own justification --
    *"a world carrying null, or a string, does [reach here]"* -- to the
    line above it.
    """
    store, world_id, session_id = derived_world

    # A second session, so the sort actually compares something.
    store.write_session(Session(session_id="s1", world_id=world_id,
                                started_at=5.0, ended_at=6.0))
    store.write_world(World(world_id=world_id, created_at=1.0, updated_at=7.0,
                            session_ids=(session_id, "s1")))
    assert build_world_listing(store)["world_count"] == 1

    path = store.session_dir(world_id, "s1") / "session.json"
    for bad in ("2026-09-11T09:00:00Z", None, [], {"a": 1}, float("nan")):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["started_at"] = bad
        path.write_text(json.dumps(record), encoding="utf-8")
        listing = build_world_listing(store)
        assert listing["world_count"] == 1, (
            f"a session whose started_at is {bad!r} emptied the whole listing"
        )


def test_an_infinity_is_not_the_newest_world(derived_world):
    """`float('inf')` sorted ahead of every real world.

    The first version of `_sortable` excluded NaN with `value == value`,
    which admits the infinities -- so an `updated_at` of `Infinity` took
    the top row under `reverse=True`, which is the exact outcome the
    function was written to stop a STRING producing. Reachable from the
    Tower's own writer: `json.dumps` emits the bare `Infinity` token by
    default and `json.loads` reads it back.
    """
    store, world_id, _ = derived_world
    original = json.loads((store.world_dir(world_id) / "world.json").read_text())

    for bad in (float("inf"), float("-inf")):
        other = dict(original, world_id="w1", session_ids=[], updated_at=bad)
        path = store.world_dir("w1") / "world.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(other))
        listing = build_world_listing(store)
        assert listing["world_count"] == 2, bad
        assert listing["worlds"][0]["world_id"] == world_id, (
            f"a world whose updated_at is {bad!r} took the top row"
        )


def test_a_manifest_is_judged_by_what_the_reader_needs_of_it(derived_world):
    """Identity and figures are different questions with different rules.

    `SCHEMA_VERSION` versions the WHOLE record family -- `World`,
    `Session`, `Keyframe`, `KeyframeEdge` -- and the manifest inherits
    `world.schema_version` rather than the module constant. The rows in
    `poses.json` and `points.json` carry no version at all; their shape is
    pinned by `world_builder.geometry/2026-08-25`, which has never moved
    under it.

    So refusing a PLACEMENT because the record schema moved is refusing a
    Sim3 for a reason that has nothing to do with it. A reviewer bumped
    the constant against the real 163-world root and watched **408
    placements across 10 sessions drop to 0** -- every segment a
    disconnected island, the picture the final solve exists to prevent.

    The figures are the other question, and there the schema matters:
    a manifest whose fields this build cannot vouch for must not say how
    many points there are.
    """
    from tower.world_builder.store import manifest_describing

    store, world_id, session_id = derived_world
    path = store.session_manifest_path(world_id, session_id)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 999
    path.write_text(json.dumps(manifest), encoding="utf-8")
    world_path = store.derived_manifest_path(world_id)
    world_manifest = json.loads(world_path.read_text(encoding="utf-8"))
    world_manifest["schema_version"] = 999
    world_path.write_text(json.dumps(world_manifest), encoding="utf-8")

    identity = manifest_describing(store, world_id, session_id, purpose="identity")
    figures = manifest_describing(store, world_id, session_id, purpose="figures")

    assert identity is not None, (
        "a record-schema bump refused a manifest whose digest is all the "
        "placement reader wanted"
    )
    assert identity["input_digest"] == manifest["input_digest"]
    assert figures is None, (
        "a manifest from an unknown schema was allowed to report figures"
    )


def test_the_drawable_predicate_does_not_re_admit_a_refused_manifest(derived_world):
    """The fifth reader, in the function written to stop there being one.

    `session_has_drawable_geometry` re-read the manifest with a LOOSE rule
    when its caller passed `None` -- re-admitting exactly what
    `validate_manifest` had just refused. The caller passes the strict
    answer and this substituted a looser one, so one session produced
    three answers: picker "complete", panel "Needs retry", render page
    blank.
    """
    from tower.world_builder.store import session_has_drawable_geometry

    store, world_id, session_id = derived_world
    derived = store.derived_dir(world_id) / session_id
    # The disk says nothing is there...
    (derived / "poses.json").write_text(json.dumps({"poses": []}))
    (derived / "points.json").write_text(json.dumps({"points": []}))
    # ...and a manifest this build cannot read says 500 points are.
    for path in (derived / "manifest.json", store.derived_manifest_path(world_id)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(schema_version=999, points=500, poses_positioned=5)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    assert session_has_drawable_geometry(store, world_id, session_id) is False, (
        "a manifest from an unknown schema was believed over the files"
    )
    # And the surface that calls it with no manifest agrees.
    row = build_world_listing(store)["worlds"][0]["sessions"][0]
    assert row["has_geometry"] is False


# -- the ghost lock ----------------------------------------------------


def _legacy_lock(store, world_id, pid, *, written_at):
    """A lock in the shape the 29 real ones have: a pid and nothing else."""
    import json as _json
    import os as _os

    path = store.lock_path(world_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json.dumps({"pid": pid}), encoding="utf-8")
    _os.utime(path, (written_at, written_at))
    return path


def test_a_recycled_pid_on_a_legacy_lock_is_not_a_live_builder(derived_world):
    """**The likeliest thing to go wrong at the retest, before this.**

    `_holder_is_running` reads `if created_at is None: return True`, and on
    the machine this campaign is preparing **29 of 163 worlds hold a lock
    file and not one carries `created_at`** -- every one predates the
    field. So all 29 are decided by the pid alone, and Windows recycles
    pids freely: two independent reviewers hit the same live alias within
    an hour, one on their first sample.

    What it does: `_most_relevant` prefers a world with a live lock over
    every saved world, so a fortnight-old empty world hijacks the default
    subscription and the phone is told `receiving`, 0 keyframes,
    `mapping_seconds: 1,407,085`. Reproduced end to end through a real
    Tower by a reviewer: the wearer's real walk built correctly, and then
    **at the moment finalization released its own lock** the live screen
    reverted to the ghost and said "Mapping" for the rest of the session.

    A process that started AFTER the lock file was written cannot be the
    process that wrote it. The filesystem keeps that timestamp for free,
    and it settles all 29 without deleting anything.
    """
    import os as _os

    import psutil

    store, world_id, _ = derived_world
    me = _os.getpid()
    started = psutil.Process(me).create_time()

    # THE GHOST: this pid, on a lock written a day before this process
    # existed. Whatever wrote that lock, it was not this process.
    _legacy_lock(store, world_id, me, written_at=started - 86_400)
    holder = store.lock_holder(world_id)
    assert holder is not None and holder["pid"] == me
    assert holder["alive"] is False, (
        "a pid recycled onto a fortnight-old lock was reported as a live "
        "builder, which is what tells the wearer a walk is in progress "
        "before they have taken a step"
    )
    assert build_world_listing(store)["worlds"][0]["live"] is False

    # A REAL BUILDER: same legacy shape, but the lock was written after
    # the process started, which is what actually happens -- the lock is
    # acquired milliseconds into the run. This must still read alive, or
    # the fix would be worse than the defect.
    _legacy_lock(store, world_id, me, written_at=started + 1.0)
    holder = store.lock_holder(world_id)
    assert holder["alive"] is True, (
        "a live builder holding a legacy lock was reported dead"
    )
    # NOT asserted through the listing: `_world_is_live` deliberately
    # excludes `holder["pid"] == os.getpid()`, so a lock this test process
    # holds reads not-live there however alive it is. `lock_holder` is the
    # answer under test; the listing's own rule is a separate one.

    store.lock_path(world_id).unlink()


def test_a_lock_that_cannot_be_read_is_not_an_idle_world(derived_world, monkeypatch):
    """`lock_holder` swallowed a transient read into "no lock at all".

    Every caller reads `None` as "this world is idle", so one collision on
    the LOCK file -- which `write_json_atomic` replaces like everything
    else -- reports a live walk as dead: `live: false` in the picker and
    the status channel dropping out of `receiving` for a poll. The suite
    showed it before a reviewer named it: the one test asserting a live
    session reads `receiving` failed once under full-suite load and passes
    otherwise.
    """
    import os as _os

    from tower.world_builder import store as store_module

    store, world_id, _ = derived_world
    _legacy_lock(store, world_id, _os.getpid(), written_at=None or 1.0)
    path = store.lock_path(world_id)
    real = store_module.read_json_closed

    def busy(target):
        if pathlib.Path(target) == path:
            raise PermissionError(13, "The process cannot access the file")
        return real(target)

    monkeypatch.setattr(store_module, "read_json_closed", busy)
    holder = store.lock_holder(world_id)
    assert holder is not None, (
        "a lock file that is right there was reported as no lock at all"
    )
    assert holder["unreadable"] is True
    assert holder["alive"] is False
    store.lock_path(world_id).unlink()
