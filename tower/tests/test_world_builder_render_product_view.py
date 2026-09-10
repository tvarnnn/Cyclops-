"""The world view: what a person sees when they open a saved world.

`tests/test_world_render.py` pins the PLY/PNG writers and the composition
rules; `tests/test_world_builder_render_route.py` pins the HTTP surface.
This file pins the PICTURE, and specifically the four ways it could start
lying again:

  * a point that carries a measured colour is drawn in some other colour
    (the bug that shipped for three weeks: `rgb` on disk, tab20 on screen);
  * a point that carries NO colour is given one anyway -- a segment
    colour, a neighbour's colour, a guess -- so a viewer cannot tell a
    grey wall from a missing measurement;
  * an unregistered fragment appears in the world frame, which is the one
    way this page can fabricate a room;
  * a degraded world renders as an empty box instead of saying what was
    actually reconstructed.

Everything here reads the SERVED HTML. No JavaScript runs, so anything a
test asserts is something the page states before a browser touches it,
which is the only kind of claim a server can be held to.
"""

import json
import re

import numpy as np
import pytest

from tower.world_builder.render import (
    CAPTION,
    CAPTION_BEHIND,
    NEUTRAL_POINT_COLOUR,
    PALETTE,
    PRODUCT_CAPTION,
    canvas_html,
    compose_frames,
    load_segments,
    render_html,
    subsample,
    VIEW_DIAGNOSTICS,
)
from tower.world_builder.store import WorldStore

WORLD = "w" * 32
SESSION = "s" * 32
DIGEST = "d" * 64

# Two colours a segment palette could never produce, so "the page drew the
# measured colour" is distinguishable from "the page drew tab20".
DARK = [10, 20, 30]
BRIGHT = [200, 210, 220]


# -- fixture ---------------------------------------------------------------


def _placement(index, state, **fields):
    row = {
        "schema_version": 1, "segment_index": index, "state": state,
        "rotation_wxyz": None, "translation": None, "scale": None,
        "reference_segment": None, "refusal_reason": None, "evidence": {},
        "frame_revision": 1, "input_digest": DIGEST,
    }
    row.update(fields)
    return row


def _registered(index, reference=0, translation=(0.0, 0.0, 0.0)):
    return _placement(index, "registered", rotation_wxyz=[1, 0, 0, 0],
                      translation=list(translation), scale=1.0,
                      reference_segment=reference)


def _write_world(root, *, points, placements, poses=None):
    derived = root / "worlds" / WORLD / "derived"
    session = derived / SESSION
    session.mkdir(parents=True)
    (session / "points.json").write_text(json.dumps({"points": points}))
    if poses is None:
        poses = [{"keyframe_id": f"{SESSION}:1", "segment_index": 0,
                  "status": "anchor", "degeneracy": "",
                  "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}]
    (session / "poses.json").write_text(json.dumps({"poses": poses}))
    (session / "placements.json").write_text(json.dumps({"placements": placements}))
    (derived / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "input_digest": DIGEST, "session_id": SESSION,
        "segments": 3, "points": len(points),
    }))
    return WorldStore(root)


def _point(index, xyz, rgb=None):
    row = {"segment_index": index, "xyz": list(xyz)}
    if rgb is not None:
        row["rgb"] = list(rgb)
    return row


# -- reading the page back -------------------------------------------------


def _frames(html):
    # The page escapes `<` so nothing can close the script block early;
    # this reads it back the way JSON.parse in the browser does.
    start = html.index("const FRAMES = ") + len("const FRAMES = ")
    return json.loads(html[start:html.index(";\n", start)].replace("\\u003c", "<"))


def _pal(html):
    match = re.search(r"const PAL = (\[.*?\]), UNREG = (\[.*?\]), STATS = (\{.*?\});",
                      html, re.S)
    assert match, "the page has no palette"
    return (json.loads(match.group(1)), json.loads(match.group(2)),
            json.loads(match.group(3)))


def _drawn_colours(frame):
    """[(xyz, '#rrggbb palette index') ...] in the world view, by walking the
    runs exactly as the page's draw loop does."""
    out = []
    for segment in frame["segments"]:
        xyz, i = segment["xyz"], 0
        for palette_index, length in segment["runs"]:
            for _ in range(length):
                out.append((tuple(xyz[i * 3:i * 3 + 3]), palette_index))
                i += 1
        assert i * 3 == len(xyz), "the runs do not cover the segment's points"
    return out


def _near(hex_colour, rgb, tolerance=8):
    channels = [int(hex_colour[k:k + 2], 16) for k in (1, 3, 5)]
    return all(abs(a - b) <= tolerance for a, b in zip(channels, rgb))


# -- the colour ------------------------------------------------------------


@pytest.fixture
def coloured(tmp_path):
    """Segment 0 registered with two measured colours, segment 1 registered
    with one measured and one missing, segment 2 refused."""
    store = _write_world(
        tmp_path / "data",
        points=[
            _point(0, (0.0, 0.0, 0.0), DARK),
            _point(0, (1.0, 0.0, 0.0), BRIGHT),
            _point(1, (0.0, 1.0, 0.0), DARK),
            _point(1, (0.0, 2.0, 0.0)),          # no rgb key at all
            _point(2, (0.0, 0.0, 9.0)),
            _point(2, (0.0, 0.0, 9.5)),
        ],
        placements=[_registered(0), _registered(1),
                    _placement(2, "refused", refusal_reason="the wearer stood still")],
    )
    return store, render_html(store, WORLD, SESSION)


def test_a_measured_colour_is_the_colour_that_is_drawn(coloured):
    _store, html = coloured
    palette, _unreg, _stats = _pal(html)
    world = _frames(html)[0]
    assert world["shared"] is True
    drawn = dict(_drawn_colours(world))
    assert _near(palette[drawn[(0.0, 0.0, 0.0)]], DARK)
    assert _near(palette[drawn[(1.0, 0.0, 0.0)]], BRIGHT)
    assert _near(palette[drawn[(0.0, 1.0, 0.0)]], DARK)
    # And it is NOT the debugger's palette: no tab20 entry is close to
    # either measured colour, so passing above cannot be an accident of
    # segment colouring.
    for entry in PALETTE:
        assert not _near(palette[drawn[(1.0, 0.0, 0.0)]], entry, tolerance=8)


def test_a_point_with_no_measured_colour_is_neutral_and_counted(coloured):
    _store, html = coloured
    palette, _unreg, _stats = _pal(html)
    assert palette[0] == "#%02x%02x%02x" % NEUTRAL_POINT_COLOUR
    world = _frames(html)[0]
    drawn = dict(_drawn_colours(world))
    # The one point in segment 1 with no rgb takes palette index 0 ...
    assert drawn[(0.0, 2.0, 0.0)] == 0
    # ... and nothing else does.
    assert sum(1 for index in drawn.values() if index == 0) == 1
    assert world["uncoloured"] == 1
    # The page says so in the served HTML, not only in the payload.
    assert "1 point carry no measured colour and are drawn neutral grey." \
        in world["summary"]
    assert world["summary"] in html


def test_a_fully_coloured_frame_says_so_rather_than_saying_nothing(tmp_path):
    """Silence would answer "is this a measurement or a placeholder" too,
    and answer it wrongly."""
    store = _write_world(
        tmp_path / "data",
        points=[_point(0, (float(i), 0.0, 0.0), BRIGHT) for i in range(4)],
        placements=[_registered(0)],
    )
    html = render_html(store, WORLD, SESSION)
    assert _frames(html)[0]["uncoloured"] == 0
    assert "Every point carries a measured colour." in html


def test_the_neutral_grey_is_never_a_segment_colour():
    """The fallback must not be drawn from PALETTE. If it were, a world
    with no photometry would silently be a segment view again."""
    assert tuple(NEUTRAL_POINT_COLOUR) not in {tuple(entry) for entry in PALETTE}


def test_a_world_with_no_photometry_is_all_neutral_and_says_so(tmp_path):
    store = _write_world(
        tmp_path / "data",
        points=[_point(0, (float(i), 0.0, 0.0)) for i in range(5)],
        placements=[_registered(0)],
    )
    html = render_html(store, WORLD, SESSION)
    palette, _unreg, _stats = _pal(html)
    # Only the neutral entry exists: nothing invented a colour table.
    assert palette == ["#%02x%02x%02x" % NEUTRAL_POINT_COLOUR]
    world = _frames(html)[0]
    assert {index for _xyz, index in _drawn_colours(world)} == {0}
    assert world["uncoloured"] == world["points"] == 5
    assert "None of them carry a measured colour" in world["summary"]
    assert "None of them carry a measured colour" in html


def test_colour_survives_subsampling_aligned_with_its_point(tmp_path):
    """A thinned cloud that paired point i with colour j would be a
    fabricated photograph of the room, and a plausible-looking one."""
    points = [_point(0, (float(i), 0.0, 0.0), [i, i, i]) for i in range(100)]
    store = _write_world(tmp_path / "data", points=points,
                         placements=[_registered(0)])
    segments = load_segments(store, WORLD, SESSION)
    subsample(segments, 20)
    kept = segments[0]
    assert 0 < len(kept.points) <= 25
    for xyz, rgb in zip(kept.points, kept.rgb):
        assert int(round(xyz[0])) == int(rgb[0]), "colour drifted off its point"
    assert kept.rgb_known.all()


# -- the world frame is only what was placed -------------------------------


def test_an_unregistered_segment_is_never_in_the_world_frame(coloured):
    _store, html = coloured
    frames = _frames(html)
    world = frames[0]
    assert world["shared"] is True
    assert [s["index"] for s in world["segments"]] == [0, 1]
    # Segment 2's points live at z=9; nothing at that depth is in the world.
    world_xyz = np.array([xyz for xyz, _c in _drawn_colours(world)])
    assert not np.any(np.abs(world_xyz[:, 2] - 9.0) < 0.6)
    # It is still reachable, in its own frame, labelled as unplaced.
    fragments = [f for f in frames if not f["shared"]]
    assert [f["placed_segments"] for f in fragments] == [0]
    assert "segment 2" in fragments[0]["label"]
    assert "own frame, own scale" in fragments[0]["name"]
    _palette, unregistered, stats = _pal(html)
    assert unregistered == [{"index": 2, "state": "refused",
                             "reason": "the wearer stood still",
                             "points": 2, "cameras": 0}]
    assert stats["unregistered"] == 1


def test_the_world_view_offers_only_shared_frames(coloured):
    """The dropdown filter lives in the page, so the test pins the flag it
    filters on rather than the filtering."""
    _store, html = coloured
    frames = _frames(html)
    assert [f["shared"] for f in frames] == [True, False]
    assert "mode === 'diag' || f.shared" in html


def test_the_largest_component_opens_first_and_the_others_are_counted(tmp_path):
    """Two components of similar size: one is the world, and the page says
    the other exists rather than overlaying it or hiding it."""
    store = _write_world(
        tmp_path / "data",
        points=([_point(0, (float(i), 0.0, 0.0), DARK) for i in range(12)]
                + [_point(1, (float(i), 5.0, 0.0), DARK) for i in range(10)]),
        placements=[_registered(0, reference=0), _registered(1, reference=1)],
    )
    html = render_html(store, WORLD, SESSION)
    frames = _frames(html)
    assert [f["points"] for f in frames] == [12, 10]      # largest first
    assert all(f["shared"] for f in frames)
    assert "1 other group of segments could not be placed relative to this one" \
        in frames[0]["summary"]
    assert frames[0]["summary"] in html                   # it is the opening line


# -- diagnostics stays reachable ------------------------------------------


def test_the_diagnostic_view_is_still_there_and_still_segment_coloured(coloured):
    _store, html = coloured
    # Both captions ride on the one payload, so both views are in this page.
    assert CAPTION in html and PRODUCT_CAPTION in html
    # The toggle is still there for a browser.
    assert "applyMode()" in html
    # Segment colours are still carried, one per segment, out of tab20.
    world = _frames(html)[0]
    colours = [s["colour"] for s in world["segments"]]
    assert colours == [f"rgb({r},{g},{b})" for r, g, b in PALETTE[:2]]
    assert len(set(colours)) == len(colours)
    # Frustums are still carried for every drawn camera.
    assert all("lines" in cam and len(cam["lines"]) == 8 for cam in world["cameras"])
    # And the diagnostic frame names -- the ones with the registration
    # detail in them -- are unchanged.
    assert world["name"].startswith("world (ref segment 0,")


def test_the_product_caption_keeps_every_disclaimer_the_old_one_made():
    """The caption may be rewritten to lead with the room instead of the
    technique. It may not quietly drop a disclaimer while doing it."""
    for claim in ("Sparse", "Not a surface", "not a mesh", "not metric scale"):
        assert claim.lower() in PRODUCT_CAPTION.lower(), claim


def test_a_stale_derived_tree_still_says_so(tmp_path):
    store = _write_world(
        tmp_path / "data",
        points=[_point(0, (0.0, 0.0, 0.0), DARK)],
        placements=[_registered(0)],
    )
    # A journal that does not hash to the manifest's digest.
    session = tmp_path / "data" / "worlds" / WORLD / "sessions" / SESSION
    session.mkdir(parents=True)
    (session / "keyframes.jsonl").write_text(
        json.dumps({"schema_version": 1, "keyframe_id": f"{SESSION}:1",
                    "session_id": SESSION, "segment_index": 0}) + "\n")
    html = render_html(store, WORLD, SESSION)
    assert CAPTION_BEHIND in html


# -- the degraded cases ----------------------------------------------------


def test_nothing_placed_says_what_was_reconstructed(tmp_path):
    store = _write_world(
        tmp_path / "data",
        points=[_point(0, (0.0, 0.0, 0.0), DARK), _point(1, (0.0, 0.0, 1.0))],
        placements=[_placement(0, "refused", refusal_reason="too few inliers"),
                    _placement(1, "unplaced")],
    )
    html = render_html(store, WORLD, SESSION)
    assert html.startswith("<!doctype html>")
    frames = _frames(html)
    assert frames and not any(f["shared"] for f in frames)
    # The sentence a person reads instead of an empty box.
    assert "2 segments were reconstructed, but none could be placed relative " \
           "to each other." in html
    assert "would invent a room" in html
    assert "1 refused, 1 unplaced" in html
    # And it is what the toolbar line says, not only what the panel says.
    assert "2 segments were reconstructed" in _payload_summary(html)


def _payload_summary(html):
    match = re.search(r'<div id="facts">(.*?)</div>', html, re.S)
    assert match
    return match.group(1)


def test_zero_points_renders_and_names_the_poses_that_do_exist(tmp_path):
    store = _write_world(
        tmp_path / "data", points=[], placements=[],
        poses=[{"keyframe_id": f"{SESSION}:1", "segment_index": 0,
                "status": "anchor", "degeneracy": "",
                "rotation": [1, 0, 0, 0], "translation": [0, 0, 0]},
               {"keyframe_id": f"{SESSION}:2", "segment_index": 0,
                "status": "unavailable", "degeneracy": "pure_rotation",
                "rotation": None, "translation": None}],
    )
    html = render_html(store, WORLD, SESSION)
    assert html.startswith("<!doctype html>")
    assert "None of the 1 segment in this session has triangulated points." in html
    assert "1 solved camera pose and 1 refused pose are on disk" in html
    assert "no geometry to show" in html


def test_an_empty_session_renders_without_claiming_a_failed_room(tmp_path):
    store = _write_world(tmp_path / "data", points=[], placements=[], poses=[])
    html = render_html(store, WORLD, SESSION)
    assert html.startswith("<!doctype html>")
    assert _frames(html) == []
    assert "Nothing was reconstructed for this session." in html
    assert "not a room that failed to map" in html


def test_an_empty_page_is_still_a_page():
    """The direct call the route's guard test uses, with nothing at all."""
    html = canvas_html([], [], [], "empty")
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "const FRAMES = [];" in html


# -- the CSP the Tower sends -----------------------------------------------


def test_the_page_loads_nothing_from_anywhere(coloured):
    """`routes/geometry.py` sends `default-src 'none'; script-src
    'unsafe-inline'; style-src 'unsafe-inline'`. Anything this page fetched
    would be blocked with no visible error, so the page must fetch nothing:
    the test is against the policy, not against a rendering."""
    _store, html = coloured
    for forbidden in ("<script src=", "<link ", "@import", "src=\"http",
                      "http://", "https://", "//cdn", "fetch(",
                      "XMLHttpRequest", "importScripts", "WebSocket",
                      "@font-face", "url("):
        assert forbidden not in html, forbidden
    # Exactly one inline script and one inline style, and no iframe or
    # object to smuggle a document through.
    assert html.count("<script>") == 1 and html.count("</script>") == 1
    assert html.count("<style>") == 1
    for element in ("<iframe", "<object", "<embed", "<img"):
        assert element not in html


# -- the opening mode is chosen by the SERVER ------------------------------
#
# It was read from `location.search` until an iOS reviewer pointed out that
# the only client this page has cannot supply one: iOS loads it with
# `loadHTMLString(_:baseURL: nil)`, so there is no URL and `location.search`
# is always empty, and its navigation policy cancels the page's own links so
# an in-page href could not reach a second view either. The affordance
# existed and did nothing on the device it was for.


def test_the_page_opens_in_the_product_view_by_default(coloured):
    _store, html = coloured
    assert "let mode = 'world';" in html
    # The API, not the word: the comment above the line explains why the URL
    # is not consulted, so a substring check would match its own rationale.
    assert "URLSearchParams" not in html, (
        "the opening mode must not depend on a URL the phone cannot supply"
    )


def test_the_page_can_be_asked_to_open_in_diagnostics(tmp_path):
    store = _write_world(
        tmp_path / "data",
        points=[_point(0, (float(i), 0.0, 0.0), DARK) for i in range(8)],
        placements=[_registered(0, reference=0)],
    )
    html = render_html(store, WORLD, SESSION, view=VIEW_DIAGNOSTICS)
    assert "let mode = 'diag';" in html
    # And it is the same payload, so nothing was lost by asking.
    assert CAPTION in html and PRODUCT_CAPTION in html


def test_an_unrecognised_view_opens_the_product_page(tmp_path):
    """A display mode on an unauthenticated route.

    The safe reading of a value nobody understands is "show the normal
    thing", not an error page: the wearer asked to see their world.
    """
    store = _write_world(
        tmp_path / "data",
        points=[_point(0, (float(i), 0.0, 0.0), DARK) for i in range(8)],
        placements=[_registered(0, reference=0)],
    )
    for view in ("", "PRODUCT", "diag", "../etc/passwd", "<script>", None):
        html = render_html(store, WORLD, SESSION, view=view)
        assert "let mode = 'world';" in html, view


def test_the_view_value_cannot_reach_the_page_as_script(tmp_path):
    """The mode is substituted into a <script> block, so it must never be
    the caller's string. It is one of two literals or nothing."""
    store = _write_world(
        tmp_path / "data",
        points=[_point(0, (float(i), 0.0, 0.0), DARK) for i in range(8)],
        placements=[_registered(0, reference=0)],
    )
    html = render_html(store, WORLD, SESSION, view="'; alert(1); //")
    assert "alert(1)" not in html
    assert "let mode = 'world';" in html
