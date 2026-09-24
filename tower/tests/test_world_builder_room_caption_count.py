"""Review V8, LOW-b: the room caption's "N more areas shown separately" counts every area
the wearer can open (contract WORLD-BUILDER-COMPONENTS.md §4, §8).

`_room_captions` (results/world_builder_render.py) left out every area whose photographic
word was `failed` or `unattempted` (review V6, L6: an area nobody will show is not
"shown separately"). But the word is `failed` when the APPEARANCE stage failed and
`unattempted` when it was not requested, even though the area's SURFACE was built -- and
then `has_geometry` is true, the area route answers 200, and the phone's areas row opens
it (§8). The caption said one area fewer than the row. An area is now counted when it is
drawable, whatever its word; one with nothing drawable and a settled word is still not.
"""

from __future__ import annotations

from tests.test_world_builder_components_areas import (  # noqa: F401 -- fixtures
    AREA1,
    AREA2,
    SESSION,
    WORLD,
    _client,
    area_world,
    build_area,
    old_world,
)
from tower.world_builder import components as C


def _page(world):
    return _client(world).get(f"/worlds/{WORLD}/render?viewer=appearance-1").text


def _row_openable(world):
    """What the phone's areas row can open, from the listing's own `components`."""
    comps = C.components_for_session(world.store, WORLD, SESSION)
    return sum(1 for e in comps if e["shown_as"] == "area" and e["has_geometry"])


def test_an_area_whose_appearance_failed_but_whose_surface_is_drawable_is_counted(area_world):
    """Before: "1 more area" -- AREA2 (owed) only, while the row could open AREA1 too."""
    build_area(area_world, AREA1, area_world.record, appearance=False)
    C.write_area_record(area_world.store, WORLD, SESSION, AREA1,
                        components_sha1=area_world.record.sha1, stage="appearance",
                        state="failed", detail="the appearance stage raised")
    word = C.area_photographic_state(area_world.store, WORLD, SESSION,
                                     area_world.store.read_session(WORLD, SESSION), AREA1,
                                     area_world.record.sha1)
    assert word["state"] == "failed"
    assert _row_openable(area_world) == 1
    assert "2 more areas shown separately" in _page(area_world)


def test_an_area_built_without_its_appearance_is_counted(area_world):
    """`unattempted` appearance ("not requested"), a drawable surface: shown, so counted."""
    build_area(area_world, AREA1, area_world.record, appearance=False)
    assert "2 more areas shown separately" in _page(area_world)


def test_a_failed_area_with_nothing_drawable_is_still_not_counted(area_world):
    """Review V6, L6 holds: nothing to open, a settled word -- not "shown separately"."""
    for area_id in (AREA1, AREA2):
        C.write_area_record(area_world.store, WORLD, SESSION, area_id,
                            components_sha1=area_world.record.sha1, stage="surface",
                            state="failed", detail="the surface stage raised")
    assert "more area" not in _page(area_world)


def test_the_caption_agrees_with_what_the_row_opens_or_will_open(area_world):
    build_area(area_world, AREA1, area_world.record, appearance=False)
    C.write_area_record(area_world.store, WORLD, SESSION, AREA1,
                        components_sha1=area_world.record.sha1, stage="appearance",
                        state="failed", detail="x")
    comps = C.components_for_session(area_world.store, WORLD, SESSION)
    from tower.world_builder.photographic import is_unsettled

    shown = sum(1 for e in comps if e["shown_as"] == "area"
                and (e["has_geometry"] or is_unsettled((e["photographic"] or {}).get("state"))))
    assert shown == 2
    assert f"{shown} more areas shown separately" in _page(area_world)


def test_an_old_world_still_has_no_caption(old_world):
    assert "more area" not in _page(old_world)
