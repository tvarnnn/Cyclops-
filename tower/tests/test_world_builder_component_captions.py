"""The components captions on the room and area pages (WORLD-BUILDER-COMPONENTS.md §4,
§5.4, contract v3).

- The room page of a session with at least one `shown_as: "area"` entry gains
  " · N more areas shown separately"; nothing else does, and a page of a session with
  `components: null` -- every saved world today -- is byte for byte the page it was.
- An area page says "Area k of N — not placed in the room", carries the not-to-scale
  caption in P2-PX's wording with when in the walk it was captured, reads "Face the
  area", and adds the levelling note exactly when its vertical could not be estimated.

The captions are applied to the templates at composition time by exact anchors; the
anchors are pinned here, so a template edit that moves one fails loudly. Where node is
on the host, the page's own caption code is RUN over a stand-in DOM, so what is
asserted is the text the wearer reads, not a substring of the program.
"""

from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from tests.test_world_builder_appearance import SESSION, WORLD, World, _never_redact
from tests.test_world_builder_components_areas import (
    AREA1,
    AREA2,
    _client,
    _entries,
    _finalize,
    _meta,
    build_area,
    write_components,
)

from tower.world_builder import appearance_render as AR
from tower.world_builder import components as C
from tower.world_builder import surface_render as SR


def _template(name):
    return (SR.viewer_template_path().parent / name).read_text(encoding="utf-8")


def _area_url(area_id, query=""):
    return f"/worlds/{WORLD}/areas/{SESSION}/{area_id}/render{query}"


@pytest.fixture
def old_world(tmp_path):
    world = World(tmp_path)
    world.build(redactor_factory=_never_redact)
    _finalize(world)
    return world


# ---------------------------------------------------------------------------
# a stand-in DOM, to run the page's own caption code under node
# ---------------------------------------------------------------------------

_DOM = r"""
function mk(tag){ return {tag, textContent: "", children: [], style: {}, hidden: false,
  className: "", append(...c){ this.children.push(...c); }, setAttribute(){} }; }
const document = {createElement: mk};
const CAP = mk("div");
function text(n){ return typeof n === "string" ? n
  : (n.textContent || "") + n.children.map(text).join(""); }
"""


def _node():
    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this host to run the page's caption code")
    return node


def _run(program: str, tmp_path) -> dict:
    path = tmp_path / "caption.js"
    path.write_text(program, encoding="utf-8")
    done = subprocess.run([_node(), str(path)], capture_output=True, text=True,
                          encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def _appearance_caption(page: str, tmp_path, *, raw=False) -> dict:
    """`updateCaption()` from the page, run: the head line and the whole caption."""
    start = page.index("  function updateCaption(){")
    end = page.index("  /* -------- verification hooks", start)
    source = page[start:end]
    button = page[page.index('<button id="bBack"'):]
    button = button[button.index(">") + 1:button.index("</button>")]
    program = (_DOM + f"""
const $ = () => CAP;
const RAW_IMAGERY = {'true' if raw else 'false'};
const manifest = {{quality: "final"}}; const S = {{layers: 39, keyframesInManifest: 39}};
const CONFIG = {{scale_state: "unknown", current: true}}; const encoding = "astc-6x6-rgba";
const NAV = {{MAX_SKIP: 3}}; let captionOpen = false;
""" + source + """
updateCaption();
console.log(JSON.stringify({head: CAP.children[0].textContent, all: text(CAP),
                            button: """ + json.dumps(button) + """}));
""")
    return _run(program, tmp_path)


def _surface_caption(page: str, tmp_path) -> dict:
    start = page.index("  const used = CONFIG.frames_used")
    end = page.index("\n", page.index("  cap.append(head", start))
    source = page[start:end]
    program = (_DOM + """
document.getElementById = () => CAP;
const CONFIG = {frames_used: 40, frames_offered: 55, faces: 1000, quality: "final",
                current: true, evidence_filter: true, scale_state: "unknown"};
const mesh = {nI: 3000};
""" + source + """
console.log(JSON.stringify({head: CAP.children[0].textContent, all: text(CAP)}));
""")
    return _run(program, tmp_path)


# ---------------------------------------------------------------------------
# the anchors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", [AR.ANCHOR_HEAD, AR.ANCHOR_WHAT_REDACTED, AR.ANCHOR_WHAT_RAW,
                                    AR.ANCHOR_FACE_TEXT, AR.ANCHOR_FACE_BUTTON])
def test_every_appearance_anchor_is_in_the_template_exactly_once(anchor):
    assert _template(AR.TEMPLATE_NAME).count(anchor) == 1


@pytest.mark.parametrize("anchor", [SR.SURFACE_ANCHOR_HEAD, SR.SURFACE_ANCHOR_BITS,
                                    SR.SURFACE_ANCHOR_NOTE])
def test_every_surface_anchor_is_in_the_template_exactly_once(anchor):
    assert _template(SR.TEMPLATE_NAME).count(anchor) == 1


@pytest.mark.parametrize("captions", [None, {}, {"more_areas": 0}])
def test_no_areas_touches_neither_template(captions):
    for name, apply in ((AR.TEMPLATE_NAME, AR.apply_captions),
                        (SR.TEMPLATE_NAME, SR.apply_surface_captions)):
        template = _template(name)
        assert apply(template, captions) == template


def test_a_moved_anchor_costs_the_caption_never_the_page():
    template = _template(AR.TEMPLATE_NAME).replace(AR.ANCHOR_FACE_TEXT, "moved")
    assert AR.apply_captions(template, {"area": {"number": 1, "of": 1}}) == template


# ---------------------------------------------------------------------------
# old worlds: byte for byte
# ---------------------------------------------------------------------------


def test_an_old_worlds_appearance_page_is_the_page_it_was(old_world):
    """Composed the way it was composed before captions existed: the template, its
    CSP and its configuration, and nothing else."""
    from tower.results.world_builder_render import build_world_render

    page = build_world_render(old_world.store, WORLD, SESSION, viewer="appearance-1")
    template = _template(AR.TEMPLATE_NAME)
    config = AR.build_appearance_config(old_world.store, WORLD, SESSION)
    config["appearance_revision"] = _config_of(page)["appearance_revision"]
    before = (template.replace(AR.TOKEN_CSP, AR.content_security_policy(AR.TRANSPORT_APP))
              .replace(AR.TOKEN_CONFIG, SR.js_object_literal(config)))
    revision = _meta(page, "wb-revision")
    stamped = before.replace('<meta name="wb-representation" content="appearance">',
                             '<meta name="wb-representation" content="appearance">'
                             f'<meta name="wb-revision" content="{revision}">', 1)
    assert page == stamped
    assert "more area" not in page and "Face the room</button>" in page


def test_an_old_worlds_surface_page_is_the_page_it_was(old_world):
    from tower.results.world_builder_render import build_world_render

    page = build_world_render(old_world.store, WORLD, SESSION, representation="surface")
    plain = SR.build_surface_page(old_world.store, WORLD, SESSION)
    assert page.replace(f'<meta name="wb-revision" content="{_meta(page, "wb-revision")}">',
                        "") == plain
    assert SR.SURFACE_ANCHOR_HEAD in page and "more area" not in page


def _config_of(page):
    import re

    m = re.search(r"const CONFIG = (\{.*?\});\n", page)
    return json.loads(m.group(1))


# ---------------------------------------------------------------------------
# §4: the room caption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("areas,expected", [(2, "2 more areas shown separately"),
                                            (1, "1 more area shown separately")])
def test_the_room_caption_counts_the_areas(old_world, tmp_path, areas, expected):
    entries = _entries(old_world.kids)
    if areas == 1:
        entries = [e for e in entries if e["id"] != AREA2]
    write_components(old_world.store, entries)
    client = _client(old_world)
    page = client.get(f"/worlds/{WORLD}/render?viewer=appearance-1").text
    caption = _appearance_caption(page, tmp_path)
    assert caption["head"] == "Captured images on reconstructed geometry · " + expected
    assert caption["button"] == "Face the room"
    surface = client.get(f"/worlds/{WORLD}/render?representation=surface").text
    got = _surface_caption(surface, tmp_path)
    assert got["all"].startswith("Reconstructed surface · ")
    assert (" · " + expected) in got["all"]


def test_the_research_room_caption_keeps_its_warning(old_world, tmp_path):
    write_components(old_world.store, _entries(old_world.kids))
    page = _client(old_world).get(f"/worlds/{WORLD}/render?viewer=appearance-1").text
    caption = _appearance_caption(page, tmp_path, raw=True)
    assert caption["head"].startswith("RESEARCH BUILD — unredacted")
    assert caption["head"].endswith(" · 2 more areas shown separately")


# ---------------------------------------------------------------------------
# §5.4: the area page
# ---------------------------------------------------------------------------


@pytest.fixture
def area_world(old_world):
    old_world.record = write_components(old_world.store, _entries(old_world.kids))
    return old_world


def test_the_area_appearance_page_says_what_it_is(area_world, tmp_path):
    build_area(area_world, AREA1, area_world.record)
    page = _client(area_world).get(_area_url(AREA1)).text
    caption = _appearance_caption(page, tmp_path)
    # AREA1 is the first of two areas (§2.4 rule 2); captured 86.7 s .. 109.6 s.
    assert caption["head"] == "Area 1 of 2 — not placed in the room"
    assert ("The camera's own images, faces redacted, from 1:26 to 1:49 of this walk. "
            "This area could not be placed relative to the room, so it is shown on its "
            "own: its position, direction and size are not comparable with the room's. "
            "Not to scale.") in caption["all"]
    assert "Scale is unknown, so distances are relative." in caption["all"]
    assert "“Face the area” turns you" in caption["all"]
    assert caption["button"] == "Face the area"
    # Every user-visible "Face the room" (a code comment keeps its own wording).
    assert "Face the room</button>" not in page and "“Face the room”" not in page
    assert "could not be estimated" not in caption["all"]   # it was levelled
    # The metas stay where the phone reads them.
    head = page[:4096]
    assert _meta(head, "wb-area") == AREA1 and _meta(head, "wb-revision")
    assert _meta(head, "wb-representation") == "appearance"


def test_an_area_that_could_not_be_levelled_says_so(area_world, tmp_path):
    build_area(area_world, AREA2, area_world.record)
    C.write_area_record(area_world.store, WORLD, SESSION, AREA2,
                        components_sha1=area_world.record.sha1, levelled=False)
    page = _client(area_world).get(_area_url(AREA2)).text
    caption = _appearance_caption(page, tmp_path)
    assert caption["head"] == "Area 2 of 2 — not placed in the room"
    assert "from 0:20 to 0:25 of this walk" in caption["all"]
    assert ("Not to scale. Its vertical could not be estimated, so it may look tilted."
            in caption["all"])
    assert _config_of(page)["area"] == {"id": AREA2, "levelled": False}


def test_the_research_area_page_never_claims_redaction(area_world, tmp_path):
    build_area(area_world, AREA1, area_world.record)
    page = _client(area_world).get(_area_url(AREA1)).text
    caption = _appearance_caption(page, tmp_path, raw=True)
    assert caption["head"] == ("RESEARCH BUILD — unredacted captured images · "
                               "Area 1 of 2 — not placed in the room")
    assert "The camera's own frames, UNREDACTED, from 1:26 to 1:49" in caption["all"]
    assert "faces redacted" not in caption["all"]
    assert "not safe to share" in caption["all"]


def test_the_area_surface_page_says_what_it_is(area_world, tmp_path):
    build_area(area_world, AREA1, area_world.record, appearance=False)
    response = _client(area_world).get(_area_url(AREA1))
    assert response.headers["content-security-policy"] == SR_STRICT
    page = response.text
    caption = _surface_caption(page, tmp_path)
    assert caption["head"] == "Area 1 of 2 — not placed in the room"
    assert ("Surfaces the Tower reconstructed from the walk, from 1:26 to 1:49 of this walk. "
            "This area could not be placed relative to the room") in caption["all"]
    assert "Not to scale." in caption["all"]
    assert "Scale is unknown, so distances here are relative." in caption["all"]
    assert _meta(page[:4096], "wb-area") == AREA1
    assert _meta(page[:4096], "wb-representation") == "surface"


SR_STRICT = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"


def test_the_captioned_pages_still_parse(area_world, tmp_path):
    """Every inline program of every captioned page is valid JavaScript."""
    import re

    build_area(area_world, AREA1, area_world.record)
    client = _client(area_world)
    pages = [client.get(_area_url(AREA1)).text,
             client.get(f"/worlds/{WORLD}/render?viewer=appearance-1").text]
    node = _node()
    for n, page in enumerate(pages):
        for m, script in enumerate(re.findall(r"<script>(.*?)</script>", page, re.S)):
            path = tmp_path / f"p{n}_{m}.js"
            path.write_text(script, encoding="utf-8")
            done = subprocess.run([node, "--check", str(path)], capture_output=True, text=True,
                                  encoding="utf-8", timeout=60)
            assert done.returncode == 0, done.stderr


def test_walk_time_is_mss_without_hours():
    assert SR.walk_time(86.7) == "1:26"
    assert SR.walk_time(3665.2) == "61:05"
    assert SR.walk_time(0) == "0:00"
