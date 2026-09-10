"""Segment placements: registration results, persisted and served.

Registration has existed as an offline script that "reads a world and
writes nothing into it". The served contract carried `registered: false`
and `transform_to_world: null` as literals, so a world with a genuine
registered cluster looked identical on the wire to one with none.

That mattered once the cluster became real: on capture `2e6cffa2`
registration places 3 of 29 segments carrying 1,917 of 4,317 points (44%),
and on `e1c52b9f` 3 of 10 carrying 5,603 of 22,520 (25%). None of it was
visible to anything downstream.

THE CACHE TRAP THIS ALSO CLOSES. `content_hash` deliberately covers only
poses and points, and the geometry-revision rollup is a hash of content
hashes. So a segment that gains a placement without its points changing
keeps its hash, and a client holding a cached chunk would never refetch --
drawing an unplaced version of a segment the world now knows how to place.
`placement_hash` exists so that state is expressible; a client keys its
cache on the pair.
"""

import pathlib
import pytest

from tower.world_builder.records import SegmentPlacement
from tower.world_builder.schema import SCHEMA_VERSION
from tower.world_builder.store import WorldStore


def _placement(**over):
    base = dict(
        segment_index=0,
        state="registered",
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation=(0.0, 0.0, 0.0),
        scale=1.0,
        reference_segment=0,
        refusal_reason=None,
        evidence={"reciprocity": 0.97},
        input_digest="x",
    )
    base.update(over)
    return SegmentPlacement(**base)


def test_placements_round_trip(tmp_path):
    store = WorldStore(tmp_path)
    world_id, session_id = "w" * 32, "s" * 32
    rows = [
        _placement(segment_index=0),
        _placement(
            segment_index=1,
            state="refused",
            rotation_wxyz=None,
            translation=None,
            scale=None,
            reference_segment=None,
            refusal_reason="the wearer stood still",
            evidence={"span_over_depth": 0.02},
        ),
    ]
    store.write_placements(world_id, session_id, rows)
    read = store.read_placements(world_id, session_id)

    assert read is not None
    assert [r.segment_index for r in read] == [0, 1]
    assert read[0].state == "registered"
    assert read[0].scale == 1.0
    assert read[1].state == "refused"
    assert read[1].scale is None
    assert read[1].refusal_reason == "the wearer stood still"


def test_absent_placements_are_absent_not_an_error(tmp_path):
    """Every world built before this existed has no placements file, and
    a reconstruction is complete without one. Absent must never be an
    error, and never a reason to refuse poses and points."""
    store = WorldStore(tmp_path)
    assert store.read_placements("w" * 32, "s" * 32) is None


def test_unreadable_placements_are_treated_as_absent(tmp_path):
    """Same rule support.json follows. A truncated index beside the
    reconstruction does not make the reconstruction wrong, and refusing
    the world over it would turn an optional file into a hard dependency
    through the back door."""
    store = WorldStore(tmp_path)
    world_id, session_id = "w" * 32, "s" * 32
    store.write_placements(world_id, session_id, [_placement()])
    path = store.derived_dir(world_id) / session_id / "placements.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert store.read_placements(world_id, session_id) is None


def test_a_refused_placement_cannot_carry_a_transform():
    """A refusal that shipped a transform would be drawn. The record
    refuses to represent that state at all."""
    with pytest.raises(ValueError, match="refused"):
        SegmentPlacement(
            segment_index=0,
            state="refused",
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation=(0.0, 0.0, 0.0),
            scale=1.0,
            reference_segment=0,
            refusal_reason="whatever",
            evidence={},
        )


def test_a_registered_placement_must_carry_a_complete_transform():
    """Half a Sim3 is not a placement. A missing scale would default to
    something somewhere, and that somewhere would be wrong."""
    with pytest.raises(ValueError, match="registered"):
        SegmentPlacement(
            segment_index=0,
            state="registered",
            rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            translation=(0.0, 0.0, 0.0),
            scale=None,
            reference_segment=0,
            refusal_reason=None,
            evidence={},
        )


def test_state_vocabulary_is_closed():
    """Consumers switch on it, so an unknown state must not reach disk."""
    with pytest.raises(ValueError, match="state"):
        SegmentPlacement(
            segment_index=0,
            state="probably",
            rotation_wxyz=None,
            translation=None,
            scale=None,
            reference_segment=None,
            refusal_reason=None,
            evidence={},
        )


def test_scale_must_be_positive_and_finite():
    """A zero scale collapses a segment to a dot at another's origin --
    the failure the registration research calls out as invisible in every
    aggregate metric. It must not be storable."""
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="scale"):
            SegmentPlacement(
                segment_index=0,
                state="registered",
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                translation=(0.0, 0.0, 0.0),
                scale=bad,
                reference_segment=0,
                refusal_reason=None,
                evidence={},
            )


# ---------------------------------------------------------------------------
# The served contract.
# ---------------------------------------------------------------------------


def _tiny_world(store, world_id="w" * 32, session_id="s" * 32):
    """A two-segment derived tree, written directly."""
    from tower.world_builder.records import CameraIntrinsics, ScaleState, World

    world = World(
        world_id=world_id,
        created_at=0.0,
        updated_at=0.0,
        display_name=None,
        scale=ScaleState(state="unknown"),
        session_ids=(session_id,),
    )
    store.write_world(world)
    poses = [
        {"keyframe_id": "k0", "segment_index": 0, "status": "anchor",
         "degeneracy": "", "rotation": None, "translation": None},
        {"keyframe_id": "k1", "segment_index": 1, "status": "anchor",
         "degeneracy": "", "rotation": None, "translation": None},
    ]
    points = [
        {"segment_index": 0, "xyz": [0.0, 0.0, 1.0]},
        {"segment_index": 1, "xyz": [1.0, 0.0, 1.0]},
    ]
    store.write_derived(
        world_id, session_id, poses=poses, points=points,
        manifest={
            # `schema_version`, because every manifest `write_derived`
            # actually writes carries one and `derived_currency` has
            # always required it. Without it this stub was refused by
            # `manifest_for` once that reader was held to the same
            # schema rule as the status channel -- and the refusal was
            # right: a manifest whose field meanings this build cannot
            # vouch for is not evidence for anybody. The stub was
            # under-specified, not the rule too strict.
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "input_digest": "x",
        },
    )
    return world_id, session_id


def test_an_unplaced_segment_is_distinguishable_from_a_refused_one(tmp_path):
    """`registered: false` alone conflated 'we tried and the two solves
    disagreed' with 'nobody looked'. On the real corpus the refusal is
    usually 'the wearer stood still', which is a message about how to
    walk -- losing it loses the only actionable part."""
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)

    manifest = adapter.build_manifest(store, world_id, session_id)
    assert manifest["segments"][0]["registration_state"] == "unplaced"

    store.write_placements(world_id, session_id, [
        _placement(
            segment_index=0, state="refused", rotation_wxyz=None,
            translation=None, scale=None, reference_segment=None,
            refusal_reason="the wearer stood still", evidence={},
        ),
    ])
    manifest = adapter.build_manifest(store, world_id, session_id)
    row = manifest["segments"][0]
    assert row["registration_state"] == "refused"
    assert row["registered"] is False
    assert row["registration_refusal_reason"] == "the wearer stood still"


def test_a_registered_segment_serves_its_transform(tmp_path):
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)
    # The reference registers ITSELF with identity, as the real writer
    # does. A registered row whose reference is not itself placed is
    # refused -- a cluster missing its origin is worse than no cluster.
    store.write_placements(world_id, session_id, [
        _placement(segment_index=0, scale=2.5, reference_segment=1),
        _placement(segment_index=1, scale=1.0, reference_segment=1),
    ])

    row = adapter.build_manifest(store, world_id, session_id)["segments"][0]
    assert row["registered"] is True
    assert row["registration_state"] == "registered"
    assert row["transform_to_world"]["scale"] == 2.5
    assert row["transform_to_world"]["reference_segment"] == 1

    chunk = adapter.build_segment(store, world_id, session_id, 0)
    assert chunk["registered"] is True
    assert chunk["transform_to_world"]["scale"] == 2.5


def test_a_placement_change_invalidates_the_cache(tmp_path):
    """THE TRAP THIS CLOSES.

    `content_hash` covers poses and points only, so a segment that gains
    a placement keeps its content hash -- by design, so cached geometry
    stays valid. That is safe ONLY because placement_hash changes
    instead. Without it a client would hold a cached chunk forever and
    draw an unplaced version of a segment the world knows how to place,
    and the old test suite could not catch it because the fields it
    checked were literals.
    """
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)

    before = adapter.build_manifest(store, world_id, session_id)
    store.write_placements(world_id, session_id, [
        _placement(segment_index=0, scale=2.5),
    ])
    after = adapter.build_manifest(store, world_id, session_id)

    row_before, row_after = before["segments"][0], after["segments"][0]
    assert row_before["content_hash"] == row_after["content_hash"], (
        "the geometry did not move, so its content hash must not"
    )
    assert row_before["placement_hash"] != row_after["placement_hash"], (
        "the placement DID move, so a client must be told"
    )
    assert before["geometry_revision"] != after["geometry_revision"], (
        "a placement-only change must move the rollup, or nothing "
        "downstream ever refetches"
    )


def test_the_rollup_still_moves_when_only_geometry_changes(tmp_path):
    """The other direction, so the rollup is not merely placement-sensitive."""
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)
    before = adapter.build_manifest(store, world_id, session_id)

    store.write_derived(
        world_id, session_id,
        poses=[{"keyframe_id": "k0", "segment_index": 0, "status": "anchor",
                "degeneracy": "", "rotation": None, "translation": None}],
        points=[{"segment_index": 0, "xyz": [9.0, 9.0, 9.0]}],
        manifest={
            # `schema_version`, because every manifest `write_derived`
            # actually writes carries one and `derived_currency` has
            # always required it. Without it this stub was refused by
            # `manifest_for` once that reader was held to the same
            # schema rule as the status channel -- and the refusal was
            # right: a manifest whose field meanings this build cannot
            # vouch for is not evidence for anybody. The stub was
            # under-specified, not the rule too strict.
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "input_digest": "x",
        },
    )
    after = adapter.build_manifest(store, world_id, session_id)
    assert before["geometry_revision"] != after["geometry_revision"]


# ---------------------------------------------------------------------------
# Transform shape and finiteness.
#
# The only value originally checked was `scale`. Seven distinct malformed
# placements reached disk and would have been served and drawn -- a
# renderer applies a NaN translation just as readily as a real one, it
# simply puts the geometry nowhere in particular. This record is the last
# place that can refuse them before they reach a screen.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,override",
    [
        ("nan in translation", {"translation": (float("nan"), 0.0, 0.0)}),
        ("inf in rotation", {"rotation_wxyz": (float("inf"), 0.0, 0.0, 0.0)}),
        ("non-unit quaternion", {"rotation_wxyz": (5.0, 5.0, 5.0, 5.0)}),
        ("rotation of length 3", {"rotation_wxyz": (1.0, 0.0, 0.0)}),
        ("translation of length 2", {"translation": (0.0, 0.0)}),
        ("scale is a bool", {"scale": True}),
        ("negative segment index", {"segment_index": -5}),
    ],
)
def test_a_malformed_transform_cannot_be_stored(label, override):
    with pytest.raises(ValueError):
        _placement(**override)


def test_a_valid_transform_is_unaffected():
    """The control. Every rejection above must be specific, not a gate that
    happens to refuse everything."""
    placement = _placement(
        rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
        translation=(1.0, 2.0, 3.0),
        scale=2.5,
        reference_segment=1,
    )
    assert placement.scale == 2.5
    assert placement.translation == (1.0, 2.0, 3.0)


def test_a_rotated_unit_quaternion_is_accepted():
    """A real rotation, not just the identity -- otherwise the norm check
    could be satisfied by only ever testing (1,0,0,0)."""
    import math as _math

    half = _math.radians(30.0) / 2.0
    placement = _placement(
        rotation_wxyz=(_math.cos(half), 0.0, _math.sin(half), 0.0)
    )
    assert placement.state == "registered"


def test_a_refused_placement_needs_no_transform_shape():
    """The shape checks must apply only where a transform exists, or a
    refusal could never be stored at all."""
    placement = _placement(
        state="refused",
        rotation_wxyz=None,
        translation=None,
        scale=None,
        reference_segment=None,
        refusal_reason="the wearer stood still",
    )
    assert placement.state == "refused"


def test_a_quaternion_at_serialised_precision_survives():
    """REGRESSION. The tolerance must not be tighter than the precision of
    the data it judges.

    The report serialiser rounds floats to five decimal places, which puts
    a unit quaternion about 1.9e-6 off unit. A first version of the norm
    check used 1e-6 and therefore rejected real placements read back from
    disk -- silently, because unrepresentable rows are dropped, so a
    registered world simply read as unregistered.
    """
    import math as _math

    from tower.world_builder.records import QUATERNION_NORM_TOLERANCE

    # A real rotation, rounded exactly as the writer rounds it.
    half = _math.radians(37.0) / 2.0
    exact = (_math.cos(half), 0.1, _math.sin(half), 0.03)
    norm = _math.sqrt(sum(v * v for v in exact))
    unit = tuple(v / norm for v in exact)
    rounded = tuple(round(v, 5) for v in unit)

    drift = abs(_math.sqrt(sum(v * v for v in rounded)) - 1.0)
    assert drift > 0, "the fixture must actually be perturbed by rounding"
    assert drift < QUATERNION_NORM_TOLERANCE, (
        f"serialised precision drifts {drift:.2e}, which the tolerance "
        f"{QUATERNION_NORM_TOLERANCE:.0e} must accommodate"
    )
    assert _placement(rotation_wxyz=rounded).state == "registered"


def test_the_tolerance_still_refuses_a_meaningfully_wrong_quaternion():
    """The other side: accommodating rounding must not accommodate a
    quaternion that would visibly rescale the geometry it rotates."""
    with pytest.raises(ValueError, match="unit quaternion"):
        _placement(rotation_wxyz=(1.02, 0.0, 0.0, 0.0))


def test_placements_written_by_the_registration_writer_read_back(tmp_path):
    """End to end through the real writer, which is where the rounding
    lived. Every row written must be readable; a dropped row is a
    registered world silently reading as unregistered."""
    import importlib.util
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "world_registration",
        _Path(__file__).resolve().parents[1] / "scripts" / "world_registration.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import math as _math

    half = _math.radians(41.0) / 2.0
    report = {
        "reference_segment": 3,
        "segments": [
            {
                "segment_index": 3,
                "registered": True,
                "transform_to_world": {
                    "rotation_wxyz": [_math.cos(half), 0.0, _math.sin(half), 0.0],
                    "translation": [0.5, -1.25, 2.0],
                    "scale": 1.75,
                },
                "points": 500,
            },
            {
                "segment_index": 4,
                "registered": False,
                "reason": "the wearer stood still",
                "points": 12,
            },
        ],
    }
    rows = module.placements_from_report(report)
    assert len(rows) == 2

    store = WorldStore(tmp_path)
    world_id, session_id = "w" * 32, "s" * 32
    store.write_placements(world_id, session_id, rows)
    read = store.read_placements(world_id, session_id)

    assert read is not None and len(read) == 2, (
        "every written placement must read back; a dropped row makes a "
        "registered world read as unregistered"
    )
    assert read[0].state == "registered"
    assert read[0].scale == pytest.approx(1.75)
    assert read[1].state == "refused"


# ---------------------------------------------------------------------------
# placement_hash field coverage.
#
# An adversarial review deleted EVERY field from the hash payload in turn
# -- rotation, translation, scale, reference_segment, frame_revision,
# state -- and the suite stayed green on all six.
#
# The cause was that the only cache test made one transition, unplaced ->
# registered. The payload goes from a one-key dict to a six-key dict, so
# ANY hash function distinguishes them. It proved the hash is not a
# constant and nothing else, while §7 of the contract asks a client to
# stake its cache on the hash covering everything that renders.
#
# These move ONE field at a time, registered -> registered'.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field,changed",
    [
        ("rotation_wxyz", (0.0, 1.0, 0.0, 0.0)),
        ("translation", (9.0, 9.0, 9.0)),
        ("scale", 3.25),
        ("reference_segment", 1),
        ("frame_revision", 2),
    ],
)
def test_every_rendering_field_moves_the_placement_hash(field, changed):
    """One field, one transition. A hash that omits any of these lets a
    conforming client draw a segment in the wrong place forever."""
    from tower.results.world_builder_geometry import placement_hash

    before = placement_hash(_placement())
    after = placement_hash(_placement(**{field: changed}))
    assert before != after, (
        f"changing {field} left placement_hash unmoved; a client keyed on "
        f"it would never refetch"
    )


def test_the_refusal_reason_moves_the_placement_hash():
    """It is SERVED, so it must be hashed.

    It was omitted. On the real corpus 26 of 29 segments are refused and
    shared a single placement_hash, so a re-registration that changed
    every refusal reason moved nothing a client keys on -- and the reason
    is the one actionable string on the wire ("the wearer stood still" is
    a message about how to walk).
    """
    from tower.results.world_builder_geometry import placement_hash

    refused = dict(
        state="refused", rotation_wxyz=None, translation=None, scale=None,
        reference_segment=None,
    )
    before = placement_hash(_placement(refusal_reason="stood still", **refused))
    after = placement_hash(_placement(refusal_reason="no geometry", **refused))
    assert before != after


def test_the_state_moves_the_placement_hash():
    """HONEST NOTE, verified by mutation: dropping `state` from the hash
    payload is an EQUIVALENT mutation, and this test does not kill it.

    `state` is implied by the transform fields -- the record forbids a
    refused row from carrying a transform and a registered row from
    missing one -- so two placements identical in the other six fields
    cannot differ in state. Likewise, changing the unplaced sentinel's
    state string cannot collide with a real placement, because the
    sentinel hashes a one-key payload and a real placement hashes seven.
    Both are pinned by outcome below rather than by a test that would only
    pretend to kill them.
    """
    from tower.results.world_builder_geometry import placement_hash

    registered = placement_hash(_placement())
    refused = placement_hash(
        _placement(
            state="refused", rotation_wxyz=None, translation=None,
            scale=None, reference_segment=None, refusal_reason="stood still",
        )
    )
    assert registered != refused


def test_the_unplaced_sentinel_is_distinct_from_every_real_placement():
    """`None` must not collide with a stored row, or an unplaced segment
    and a placed one would share a cache key."""
    from tower.results.world_builder_geometry import placement_hash

    unplaced = placement_hash(None)
    assert unplaced != placement_hash(_placement())
    assert unplaced != placement_hash(
        _placement(
            state="refused", rotation_wxyz=None, translation=None,
            scale=None, reference_segment=None, refusal_reason="x",
        )
    )


def test_a_dropped_row_does_not_take_its_neighbours_with_it(tmp_path):
    """The commit claims a bad row is dropped and its neighbours are still
    good. That claim was untested -- a mutant discarding the neighbours
    too passed."""
    import json as _json

    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)
    store.write_placements(world_id, session_id, [
        _placement(segment_index=0, scale=2.5, reference_segment=0),
    ])
    path = store.derived_dir(world_id) / session_id / "placements.json"
    document = _json.loads(path.read_text(encoding="utf-8"))
    document["placements"].append({"segment_index": 1, "state": "nonsense"})
    path.write_text(_json.dumps(document), encoding="utf-8")

    rows = store.read_placements(world_id, session_id)
    assert [r.segment_index for r in rows] == [0], (
        "the good row must survive its bad neighbour"
    )
    served = adapter.build_manifest(store, world_id, session_id)["segments"][0]
    assert served["registered"] is True


# ---------------------------------------------------------------------------
# Staleness, corruption, and reference consistency.
# ---------------------------------------------------------------------------


def test_a_placement_from_a_different_build_is_not_served(tmp_path):
    """THE WORST DEFECT THIS CLOSES.

    write_derived rewrites poses and points wholesale and never touches
    placements.json, so a Sim3 outlives the reconstruction it was fitted
    to. Measured before the fix: after a full rebuild with completely
    different points and a new input_digest, the SAME transform was still
    served as `registered: true`.

    The cache guarantee does not rescue it -- content_hash moves, the
    client refetches exactly as it should, and is handed a transform
    solved against points that no longer exist.
    """
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)
    store.write_placements(world_id, session_id, [
        _placement(segment_index=0, scale=7.5, reference_segment=0),
    ])
    assert adapter.build_manifest(
        store, world_id, session_id
    )["segments"][0]["registered"] is True

    # A full rebuild: different points, different digest.
    store.write_derived(
        world_id, session_id,
        poses=[{"keyframe_id": "k0", "segment_index": 0, "status": "anchor",
                "degeneracy": "", "rotation": None, "translation": None}],
        points=[{"segment_index": 0, "xyz": [42.0, -17.0, 999.0]}],
        manifest={"session_id": session_id, "input_digest": "a-new-build"},
    )

    row = adapter.build_manifest(store, world_id, session_id)["segments"][0]
    assert row["registered"] is False, (
        "a transform solved against points that no longer exist must not "
        "be served as a placement"
    )
    assert row["registration_state"] == "unplaced"
    assert row["transform_to_world"] is None


def test_a_placement_with_no_digest_cannot_be_verified_so_is_not_served(
    tmp_path,
):
    """Unbound is unverifiable, and an unverifiable transform is the
    failure this whole mechanism exists to prevent."""
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)
    store.write_placements(world_id, session_id, [
        _placement(segment_index=0, reference_segment=0, input_digest=None),
    ])
    row = adapter.build_manifest(store, world_id, session_id)["segments"][0]
    assert row["registered"] is False
    assert row["registration_state"] == "unplaced"


def test_a_registered_row_whose_reference_is_unplaced_is_refused(tmp_path):
    """A cluster missing its origin is worse than no cluster.

    The contract says segments sharing a reference_segment are in one
    space and may be drawn together. A registered row pointing at a
    segment that is absent, dropped, or itself refused invites a client to
    composite geometry into a frame nothing defines.
    """
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)
    store.write_placements(world_id, session_id, [
        _placement(segment_index=0, scale=2.5, reference_segment=1),
        # segment 1, the reference, is REFUSED rather than registered
        _placement(
            segment_index=1, state="refused", rotation_wxyz=None,
            translation=None, scale=None, reference_segment=None,
            refusal_reason="the wearer stood still",
        ),
    ])
    rows = adapter.build_manifest(store, world_id, session_id)["segments"]
    assert rows[0]["registered"] is False, (
        "a segment placed against an unplaced reference must not claim a "
        "shared frame"
    )


@pytest.mark.parametrize(
    "label,body",
    [
        ("truncated json", "{ this is not json"),
        ("placements is null", '{"placements": null}'),
        ("top-level list", "[]"),
        ("placements is a string", '{"placements": "oops"}'),
        ("a row is null", '{"placements": [null]}'),
        ("missing key", '{"other": 1}'),
    ],
)
def test_no_corruption_shape_can_break_the_geometry_channel(
    tmp_path, label, body
):
    """`read_placements` promises it never raises. It caught only invalid
    JSON syntax; four structurally-VALID shapes raised straight through
    build_manifest into an HTTP 500, taking poses and points that were
    perfectly good down with an optional index beside them.

    The original test used the single shape that happened to work.
    """
    from tower.results import world_builder_geometry as adapter

    store = WorldStore(tmp_path)
    world_id, session_id = _tiny_world(store)
    store.write_placements(world_id, session_id, [
        _placement(segment_index=0, reference_segment=0),
    ])
    path = store.derived_dir(world_id) / session_id / "placements.json"
    path.write_text(body, encoding="utf-8")

    manifest = adapter.build_manifest(store, world_id, session_id)
    assert manifest is not None and manifest["segment_count"] == 2, (
        f"{label} broke the geometry channel"
    )
    chunk = adapter.build_segment(store, world_id, session_id, 0)
    assert chunk is not None


def test_non_standard_json_never_reaches_disk(tmp_path):
    """Python writes NaN and Infinity as bare tokens that JSON.parse,
    Swift's JSONSerialization and Go's encoding/json all reject. A file
    only this runtime can read is not a wire artifact."""
    store = WorldStore(tmp_path)
    world_id, session_id = "w" * 32, "s" * 32
    with pytest.raises(ValueError):
        store.write_placements(world_id, session_id, [
            _placement(evidence={"reciprocity": float("nan")}),
        ])
    path = store.derived_dir(world_id) / session_id / "placements.json"
    assert not path.exists(), "a refused write must leave no file"


def test_a_registered_placement_cannot_carry_a_refusal_reason():
    """The wire drops it silently for a registered row, so storing one
    creates a field that looks meaningful and is not."""
    with pytest.raises(ValueError, match="refusal reason"):
        _placement(refusal_reason="bogus")


@pytest.mark.parametrize(
    "label,body",
    [
        ("truncated json", "{ not json"),
        ("support is null", '{"support": null}'),
        ("top-level list", "[]"),
        ("support is a string", '{"support": "oops"}'),
    ],
)
def test_no_support_corruption_shape_breaks_or_leaks(tmp_path, label, body):
    """`_read_support` promises it never raises. It raised on a top-level
    list -- and, worse, RETURNED the string "oops" as the support table on
    another shape, which cross-segment registration then consumes.

    Same class of defect as the placements reader had; the new code had
    copied the old shape.
    """
    store = WorldStore(tmp_path)
    world_id, session_id = "w" * 32, "s" * 32
    derived = store.derived_dir(world_id) / session_id
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "poses.json").write_text('{"poses": []}', encoding="utf-8")
    (derived / "points.json").write_text('{"points": []}', encoding="utf-8")
    (derived / "support.json").write_text(body, encoding="utf-8")

    read = store.read_derived(world_id, session_id, verify=False)
    assert read is not None, f"{label} must not refuse the whole world"
    assert read["support"] is None, (
        f"{label} must read as absent, never as a usable support table"
    )


def test_a_valid_support_table_still_reads(tmp_path):
    """The control: the shape check must not refuse real data."""
    store = WorldStore(tmp_path)
    world_id, session_id = "w" * 32, "s" * 32
    derived = store.derived_dir(world_id) / session_id
    derived.mkdir(parents=True, exist_ok=True)
    (derived / "poses.json").write_text('{"poses": []}', encoding="utf-8")
    (derived / "points.json").write_text('{"points": []}', encoding="utf-8")
    (derived / "support.json").write_text(
        '{"support": [[0, 1, 2], [0, 3, 4]]}', encoding="utf-8"
    )
    read = store.read_derived(world_id, session_id, verify=False)
    assert read["support"] == [[0, 1, 2], [0, 3, 4]]


def test_a_manifest_being_replaced_is_not_an_http_500(derived_world, monkeypatch):
    """The Windows collision, made deterministic.

    `write_derived` replaces **five** files microseconds apart, and on
    Windows `os.replace` onto a path a reader holds open fails with
    WinError 5 -- symmetrically, so does the reader's `open()` inside the
    writer's window. Neither manifest reader caught `OSError`;
    `derived_currency` calls both with no `try`, and
    `world_builder_geometry._is_current` calls `derived_currency` outside
    its own. So a `PermissionError` walked out through `build_manifest`
    and became an HTTP **500** -- measured by a reviewer at 2.4% of
    requests beside a live writer, on a route whose own comment promises
    *"404 now means ABSENT only"* and for which the phone has no branch.

    A round of this campaign measured that hazard and added retries to
    `poses.json` and `points.json`, the two files it happened to be
    reading, and left the three the writer replaces beside them alone.

    Injecting the error is the only way to test it deterministically: a
    real collision needs two processes and a 2.4% chance.
    """
    from tower.results import world_builder_geometry as adapter
    from tower.world_builder import store as store_module

    store, world_id, session_id = derived_world
    target = store.session_manifest_path(world_id, session_id)
    real = store_module.read_json_closed
    calls = {"n": 0}

    def flaky(path):
        if pathlib.Path(path) == target:
            calls["n"] += 1
            if calls["n"] <= 2:
                raise PermissionError(13, "The process cannot access the file")
        return real(path)

    monkeypatch.setattr(store_module, "read_json_closed", flaky)

    # A transient collision is ridden out, not raised and not lost.
    manifest = adapter.build_manifest(store, world_id, session_id)
    assert manifest is not None
    assert calls["n"] >= 3, "the reader did not retry past the replace"
    assert manifest["segment_count"] >= 1


def test_a_manifest_that_stays_unreadable_is_absent_not_an_exception(
    derived_world, monkeypatch
):
    """And a fault that outlives the retries must still not reach the route.

    Returning "no manifest" is degraded and honest: every caller treats an
    absent manifest as "nothing here can judge it", the geometry is served
    with `current: false`, and the status channel says so in words. The
    raise is a 500 the phone cannot read.
    """
    from tower.results import world_builder_geometry as adapter
    from tower.world_builder import store as store_module

    store, world_id, session_id = derived_world
    real = store_module.read_json_closed
    manifests = {
        store.session_manifest_path(world_id, session_id),
        store.derived_manifest_path(world_id),
    }

    def always_busy(path):
        if pathlib.Path(path) in manifests:
            raise PermissionError(13, "The process cannot access the file")
        return real(path)

    monkeypatch.setattr(store_module, "read_json_closed", always_busy)

    manifest = adapter.build_manifest(store, world_id, session_id)
    assert manifest is not None, "a busy manifest took the geometry with it"
    assert manifest["current"] is False, (
        "currency was asserted from a manifest that could not be read"
    )
