"""The appearance page: the top rung of the render ladder, its revision, its
transport and its Content-Security-Policy.

Contract: docs/contracts/WORLD-BUILDER-WORLDS.md §4 / §4a and
docs/contracts/WORLD-BUILDER-APPEARANCE.md §9. Built on the synthetic solved,
surfaced world of `test_world_builder_appearance.py`.
"""

from __future__ import annotations

import json
import re
import sys

import pytest

from tests.test_world_builder_appearance import SESSION, WORLD, World, _app, _never_redact

APP_POLICY = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
              "connect-src glasses-world:")
DEBUG_POLICY = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                "connect-src 'self'")
STRICT_POLICY = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'"
# What the iOS app built with the appearance page declares (§4 `viewer`).
V = "appearance-1"


@pytest.fixture
def built(tmp_path):
    world = World(tmp_path)
    world.build(redactor_factory=_never_redact)
    return world


def _meta(html, name):
    m = re.search(rf'<meta name="{name}" content="([^"]*)">', html[:4096])
    return m.group(1) if m else None


def _meta_policy(html):
    m = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]*)">', html[:4096])
    return m.group(1) if m else None


def _config(html):
    m = re.search(r"const CONFIG = (\{.*?\});\n", html)
    assert m, "the page carries its configuration"
    return json.loads(m.group(1))


def _client(world):
    from fastapi.testclient import TestClient

    return TestClient(_app(world.root))


class TestTheLadder:

    def test_auto_serves_the_appearance_when_one_can_be_served(self, built):
        from tower.results.world_builder_render import build_world_render

        html = build_world_render(built.store, WORLD, SESSION, viewer=V)
        assert _meta(html, "wb-representation") == "appearance"
        assert "Captured images on reconstructed geometry" in html

    def test_the_page_and_the_revision_route_agree(self, built):
        from tower.results.world_builder_render import build_render_revision, build_world_render

        rev = build_render_revision(built.store, WORLD, SESSION, viewer=V)
        assert rev["representation"] == "appearance"
        epoch = built.manifest()["epoch"]
        assert rev["revision"] == f"{SESSION}/appearance:1@{epoch}"
        assert rev["appearance"]["epoch"] == epoch and rev["appearance"]["state"] == "served"
        html = build_world_render(built.store, WORLD, SESSION, viewer=V)
        assert _meta(html, "wb-revision") == rev["revision"]
        assert _config(html)["appearance_revision"] == rev["appearance"]["revision"]

    def test_a_new_appearance_build_does_not_move_the_page_revision(self, built):
        """The page follows builds itself; a moved page revision would make the
        phone reload it and reset the wearer's camera on every solve."""
        from tower.results.world_builder_render import build_render_revision

        first = build_render_revision(built.store, WORLD, SESSION, viewer=V)
        built.build(force=True, redactor_factory=_never_redact)
        second = build_render_revision(built.store, WORLD, SESSION, viewer=V)
        assert second["appearance"]["revision"] != first["appearance"]["revision"]
        assert second["revision"] == first["revision"]
        assert second["representation"] == "appearance"

    def test_an_appearance_behind_the_surface_is_still_the_top_rung(self, built):
        """Currency is reported, never enforced: a live walk's new surface must
        not swap the page down and back up for the minute a rebuild takes."""
        from tower.results.world_builder_render import build_render_revision, build_world_render

        built.write_surface(subdivisions=12)
        rev = build_render_revision(built.store, WORLD, SESSION, viewer=V)
        assert rev["representation"] == "appearance"
        assert rev["appearance"]["current"] is False
        config = _config(build_world_render(built.store, WORLD, SESSION, viewer=V))
        assert config["current"] is False and config["currency_reason"]

    def test_a_relabelled_session_steps_down_to_the_surface(self, built):
        from tower.results.world_builder_render import (
            WorldRenderUnavailable,
            build_render_revision,
            build_world_render,
        )

        built.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
        html = build_world_render(built.store, WORLD, SESSION, viewer=V)
        assert _meta(html, "wb-representation") == "surface"
        assert build_render_revision(built.store, WORLD, SESSION,
                                     viewer=V)["representation"] == "surface"
        with pytest.raises(WorldRenderUnavailable):
            build_world_render(built.store, WORLD, SESSION, representation="appearance")

    def test_a_purged_world_steps_down_too(self, built):
        from tower.results.world_builder_render import build_world_render

        record = built.store.read_world(WORLD)
        built.store.write_world(type(record)(**{**record.__dict__, "images_purged": True}))
        assert _meta(build_world_render(built.store, WORLD, SESSION, viewer=V),
                     "wb-representation") != "appearance"

    def test_a_world_without_appearance_gets_the_page_it_got_before(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        world = World(tmp_path)
        auto = build_world_render(world.store, WORLD, SESSION, viewer=V)
        assert _meta(auto, "wb-representation") == "surface"
        assert auto == build_world_render(world.store, WORLD, SESSION, representation="surface")

    def test_pinning_appearance_without_one_is_404(self, tmp_path):
        from tower.results.world_builder_render import WorldRenderUnavailable, build_world_render

        world = World(tmp_path)
        with pytest.raises(WorldRenderUnavailable):
            build_world_render(world.store, WORLD, SESSION, representation="appearance")

    def test_surface_is_still_reachable_on_request(self, built):
        from tower.results.world_builder_render import build_world_render

        html = build_world_render(built.store, WORLD, SESSION, representation="surface")
        assert _meta(html, "wb-representation") == "surface"

    def test_diagnostics_stays_sparse(self, built):
        from tower.results.world_builder_render import (
            WorldRenderUnavailable,
            build_render_revision,
            build_world_render,
        )

        # This synthetic world has no sparse tree, so the sparse page is absent:
        # a 404, never the appearance page in its place.
        with pytest.raises(WorldRenderUnavailable):
            build_world_render(built.store, WORLD, SESSION, view="diagnostics", viewer=V)
        assert build_render_revision(built.store, WORLD, SESSION, view="diagnostics",
                                     viewer=V)["representation"] == "sparse"

    def test_an_appearance_module_that_will_not_import_costs_nothing(self, built, monkeypatch):
        import builtins

        from tower.results.world_builder_render import build_world_render

        real_import = builtins.__import__

        def broken(name, *a, **kw):
            if name == "tower.world_builder.appearance_render":
                raise ImportError("simulated broken module")
            return real_import(name, *a, **kw)

        monkeypatch.setitem(sys.modules, "tower.world_builder.appearance_render", None)
        monkeypatch.setattr(builtins, "__import__", broken)
        html = build_world_render(built.store, WORLD, SESSION, viewer=V)
        assert _meta(html, "wb-representation") == "surface"

    def test_a_page_that_cannot_be_composed_falls_back(self, built, monkeypatch):
        from tower.results.world_builder_render import build_world_render
        from tower.world_builder import appearance_render as AR

        def refuse(*a, **kw):
            raise AR.AppearanceViewerUnavailable("simulated")

        monkeypatch.setattr(AR, "build_appearance_page", refuse)
        assert _meta(build_world_render(built.store, WORLD, SESSION, viewer=V),
                     "wb-representation") == "surface"

    def test_an_appearance_bug_is_reported_as_a_bug_not_as_a_relabel(self, built, monkeypatch):
        """Review 2, m-15. The revision route must survive an exception in the
        appearance code -- and it did -- but it reported the survival as
        `withdrawn`, which is the Tower's word for a RELABEL or a PURGE. Both
        consumers act on it as a privacy event: the page tells the wearer their
        redaction record changed, and the app records a withdrawal. A traceback
        is neither. This branch had no test at all on the Python side.
        """
        from tower.results import world_builder_appearance as WA
        from tower.results.world_builder_render import build_render_revision

        def crash(*a, **kw):
            raise RuntimeError("simulated appearance bug")

        monkeypatch.setattr(WA, "appearance_revision", crash)
        rev = build_render_revision(built.store, WORLD, SESSION, viewer=V)
        assert rev["appearance"]["state"] == "unavailable"
        assert rev["appearance"]["revision"] is None
        assert rev["appearance"]["current"] is False
        # and the page revision still answers: the rung steps down, the route
        # does not 500, and a follower keeps following.
        assert rev["representation"] in {"surface", "dense", "sparse"}
        assert isinstance(rev["revision"], str) and rev["revision"]


class TestOldApps:
    """An iOS build older than the appearance page has no `glasses-world:` scheme
    handler: it loads every page with `loadHTMLString` and could fetch none of
    the appearance imagery. It declares nothing, so `auto` must give it the
    surface page it got before the rung existed (§4 `viewer`)."""

    def test_auto_without_the_declaration_is_the_surface(self, built):
        from tower.results.world_builder_render import build_render_revision, build_world_render

        html = build_world_render(built.store, WORLD, SESSION)
        assert _meta(html, "wb-representation") == "surface"
        assert html == build_world_render(built.store, WORLD, SESSION, representation="surface")
        rev = build_render_revision(built.store, WORLD, SESSION)
        assert rev["representation"] == "surface"
        assert _meta(html, "wb-revision") == rev["revision"]
        # The appearance is still reported: it is additive data, not a rung.
        assert rev["appearance"]["revision"] is not None

    def test_only_the_named_token_declares_it(self, built):
        from tower.results.world_builder_render import (
            build_render_revision,
            build_world_render,
            viewer_draws_appearance,
        )

        for viewer in (None, "", "appearance", "appearance-2", "APPEARANCE-1", "surface-1",
                       "appearance-1x", " , "):
            assert not viewer_draws_appearance(viewer), viewer
            assert _meta(build_world_render(built.store, WORLD, SESSION, viewer=viewer),
                         "wb-representation") == "surface", viewer
            assert build_render_revision(built.store, WORLD, SESSION,
                                         viewer=viewer)["representation"] == "surface", viewer
        for viewer in ("appearance-1", "future-2,appearance-1", " appearance-1 ,x"):
            assert viewer_draws_appearance(viewer), viewer
            assert _meta(build_world_render(built.store, WORLD, SESSION, viewer=viewer),
                         "wb-representation") == "appearance", viewer

    def test_a_pinned_rung_is_served_without_the_declaration(self, built):
        """A caller naming the rung is asking for that page (a desktop debug
        session, a test); the gate is on `auto` only."""
        from tower.results.world_builder_render import build_world_render

        html = build_world_render(built.store, WORLD, SESSION, representation="appearance")
        assert _meta(html, "wb-representation") == "appearance"

    def test_the_routes_gate_on_the_query(self, built):
        client = _client(built)
        old = client.get(f"/worlds/{WORLD}/render")
        assert old.status_code == 200 and _meta(old.text, "wb-representation") == "surface"
        assert old.headers["content-security-policy"] == STRICT_POLICY
        new = client.get(f"/worlds/{WORLD}/render", params={"viewer": V})
        assert new.status_code == 200 and _meta(new.text, "wb-representation") == "appearance"
        assert client.get(f"/worlds/{WORLD}/render/revision").json()["representation"] == "surface"
        assert client.get(f"/worlds/{WORLD}/render/revision",
                          params={"viewer": V}).json()["representation"] == "appearance"
        # An unknown or odd declaration is ignored, never refused.
        odd = client.get(f"/worlds/{WORLD}/render", params={"viewer": "x" * 2000})
        assert odd.status_code == 200 and _meta(odd.text, "wb-representation") == "surface"


class TestThePage:

    def test_it_carries_no_imagery_and_fetches_only_its_own_routes(self, built):
        from tower.world_builder.appearance_render import build_appearance_page

        html = build_appearance_page(built.store, WORLD, SESSION)
        # A SHELL, NOT A DATA PAGE. The cap is on the program, and the program
        # is source and comments: the imagery is ~13 MB of blocks the page
        # fetches, so anything that carried it would be off this scale
        # entirely (the assertions below are what actually check that). 256 KB
        # held from the first page until 2026-09-21, when the fix-it bestview
        # lane added the Best view's clean term, the opening's surround term,
        # the standoff's drift, the saved camera and the status-bar fix and
        # went 14 KB over. Raised to 288 KB rather than paid for by deleting
        # the reasoning: on the wire it is gzip, where the whole page is 92 KB
        # against the 85 KB it was, and one texture chunk is 1.4 MB.
        assert len(html.encode("utf-8")) < 288 * 1024, "a shell, not a data page"
        assert "WBAPCK" not in html.split("<script>")[0]
        manifest = built.manifest()
        for chunk in manifest["chunks"]:
            assert chunk["digest"] not in html
        config = _config(html)
        assert config["base"] == "glasses-world://tower"
        assert config["routes"] == {
            "manifest": f"/worlds/{WORLD}/appearance/{SESSION}/manifest",
            "chunk": f"/worlds/{WORLD}/appearance/{SESSION}/chunk/",
            "proxy": f"/worlds/{WORLD}/appearance/{SESSION}/proxy/",
            "revision": f"/worlds/{WORLD}/render/revision?session_id={SESSION}",
        }
        # Every fetch goes through one helper that prefixes CONFIG.base.
        assert html.count("fetch(") == 1 and "fetch(url(route)" in html
        assert not re.search(r"""(src|href)\s*=\s*["'](https?:)?//""", html)
        assert "XMLHttpRequest" not in html and "WebSocket" not in html
        assert "__WB_APPEARANCE" not in html

    def test_it_says_what_it_is_and_never_claims_metres(self, built):
        from tower.world_builder.appearance_render import build_appearance_page

        html = build_appearance_page(built.store, WORLD, SESSION)
        assert "faces redacted" in html
        # It said "nothing there is filled in" until review 2 found that the
        # page had been filling thin cracks and painting voids with fog since
        # `21d6f1a`. What replaces it is a BOUND rather than a denial: the
        # wearer is told what is filled and how wide it can be.
        # And it no longer promises a "grey haze ... always darker than the room
        # around it": the last review measured that unobserved geometry rendered
        # at luminance 10-18 against a background of 11-28, so in the opening
        # view the haze was darker than the emptiness it was supposed to be
        # distinguishable from. The page draws a flat grey patch now (the
        # shader's `haze`) and the sentence says what a reader can check.
        assert "A flat grey patch is a place no kept frame saw" in html
        assert "no texture and no detail at any scale" in html
        assert "always darker than the room around it" not in html
        assert "Cracks a few pixels wide between two parts of one surface are closed" in html
        assert "nothing there is filled in" not in html
        assert "Scale is unknown" in html
        assert " metres" not in html and " meters" not in html

    def test_the_debug_transport_is_the_towers_origin_only_when_named(self, built):
        from tower.world_builder.appearance_render import build_appearance_page

        app = build_appearance_page(built.store, WORLD, SESSION)
        debug = build_appearance_page(built.store, WORLD, SESSION, transport="tower")
        odd = build_appearance_page(built.store, WORLD, SESSION, transport="https://evil.example")
        assert _meta_policy(app) == APP_POLICY and _config(app)["transport"] == "app"
        assert _meta_policy(debug) == DEBUG_POLICY and _config(debug)["base"] == ""
        assert _meta_policy(odd) == APP_POLICY and _config(odd)["base"] == "glasses-world://tower"

    def test_it_is_unshaded(self):
        from tower.world_builder.appearance_render import viewer_template_path

        text = viewer_template_path().read_text(encoding="utf-8")
        blend = text[text.index("const GLSL_SHADE"):text.index("/* ---------- main")]
        for lighting in ("dot(n,", "normalize(cross(dFdx", "hemi", "uUpView"):
            assert lighting not in blend
        # the surface's facing (the proxy's smooth vertex normal) exists only to
        # keep an oblique source from painting a surface it saw edge-on; it
        # never reaches a colour
        uses = [line for line in blend.splitlines() if "facing" in line and "//" not in line.split("facing")[0]]
        assert len(uses) == 2 and "vec3 facing = N;" in uses[0] and "float cosI = abs(dot(facing, ds));" in uses[1], uses
        cos_uses = [line for line in blend.splitlines() if "cosI" in line and "//" not in line.split("cosI")[0]]
        assert len(cos_uses) == 2 and "float edgeOn = smoothstep(0.12, 0.35, cosI);" in cos_uses[1], cos_uses

    def test_it_levels_the_horizon_and_walks_the_recorded_path(self, built):
        from tower.world_builder.appearance_render import build_appearance_config

        config = build_appearance_config(built.store, WORLD, SESSION)
        assert len(config["cameras"]) == len(built.kids)
        assert config["up"] is None or len(config["up"]) == 3

    def test_it_walks_and_levels_to_the_room_component_only(self, built):
        """Walk 1 (`ee48aae3`, 2026-09-24). A split walk's solution keeps every
        component's poses, and the room page opened on, walked and levelled to
        all of them: 103 cameras for a 73-keyframe room, the scale-mismatched
        area's 30 about 40 room radii away (its up tilted 11 degrees), and on
        `6e6d3fc3` five separately solved components in gauges of their own."""
        import dataclasses

        from tower.world_builder.appearance_render import build_appearance_config
        from tower.world_builder.surface_render import _camera_up

        room = build_appearance_config(built.store, WORLD, SESSION)
        up = _camera_up(built.store, WORLD, SESSION)
        stray = f"{SESSION}:{999:08d}"
        # another component, in another gauge: far away and rolled 90 degrees
        poses = {**built.poses, stray: {"component": 1, "observations": 50,
                                        "rotation": [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                                        "translation": [40.0, -3.0, 25.0]}}
        built._write_solution(built._workspace, dataclasses.replace(
            built.solution, keyframe_ids=[*built.kids, stray], poses=poses))
        split = build_appearance_config(built.store, WORLD, SESSION)
        assert len(split["cameras"]) == len(built.kids)
        assert split["cameras"] == room["cameras"], "the room's walk is the room's cameras"
        assert _camera_up(built.store, WORLD, SESSION) == up, "and so is its vertical"

    def test_a_solution_without_a_room_component_still_walks_and_bad_components_do_not_raise(self, built):
        """Codex review of the room-only filter (HIGH, MED): a solution whose poses
        carry no component 0 walks every pose, as before the filter, rather than
        nothing; a component that is not a number is not the room's and raises
        nothing while the page is composed."""
        import dataclasses

        from tower.world_builder.appearance_render import build_appearance_config
        from tower.world_builder.surface_render import _room_poses

        room = build_appearance_config(built.store, WORLD, SESSION)
        shifted = {kid: {**p, "component": 1} for kid, p in built.poses.items()}
        built._write_solution(built._workspace, dataclasses.replace(built.solution, poses=shifted))
        assert build_appearance_config(built.store, WORLD, SESSION)["cameras"] == room["cameras"]
        odd = {**built.poses, "x:1": {**next(iter(built.poses.values())), "component": None},
               "x:2": {**next(iter(built.poses.values())), "component": "room"}}
        kept = _room_poses(dataclasses.replace(built.solution, poses=odd))
        assert "x:1" not in kept and "x:2" not in kept and len(kept) == len(built.poses)

    def test_the_scene_unit_is_clamped(self):
        from tower.world_builder.appearance_render import viewer_template_path

        text = viewer_template_path().read_text(encoding="utf-8")
        assert "sceneUnit: (u => u > 0 && isFinite(u) ? Math.min(1e3, Math.max(1e-3, u)) : 1)" in text

    def test_the_template_is_installed_with_its_tokens(self):
        from tower.world_builder.appearance_render import TOKEN_CONFIG, TOKEN_CSP, viewer_template_path

        text = viewer_template_path().read_text(encoding="utf-8")
        assert TOKEN_CONFIG in text and TOKEN_CSP in text
        assert text.index('name="wb-representation" content="appearance"') < 400


def _template():
    from tower.world_builder.appearance_render import viewer_template_path

    return viewer_template_path().read_text(encoding="utf-8")


def _section(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]


class TestTheBlend:
    """What the page draws (WORLD-BUILDER-WORLDS.md §4), checked on its source:
    the renderer runs only in a browser, and the behaviour is measured there by
    the viewer-polish lane (Glasses-scratch/wb-final-recon/fixit/viewer-polish)."""

    def test_the_script_parses(self, built):
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            pytest.skip("no node on this host to parse the page's script")
        from tower.world_builder.appearance_render import build_appearance_page

        html = build_appearance_page(built.store, WORLD, SESSION, transport="tower")
        script = html[html.index("<script>") + len("<script>"):html.rindex("</script>")]
        r = subprocess.run([node, "--check", "-"], input=script, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr[-2000:]

    def test_the_config_carries_the_measured_blend_constants(self, built):
        from tower.world_builder import appearance_render as R

        config = R.build_appearance_config(built.store, WORLD, SESSION)
        assert config["border_feather_px"] == R.BORDER_FEATHER_PX == 80
        assert config["edge_fade_px"] == R.EDGE_FADE_PX
        assert config["blend_temperature"] == R.BLEND_TEMPERATURE
        assert config["source_fade_ms"] == R.SOURCE_FADE_MS > 0
        assert config["consensus"] == R.CONSENSUS

    def test_no_test_switches_a_source_on_or_off_between_neighbouring_pixels(self):
        blend = _section(_template(), "const GLSL_SHADE", "/* ---------- main")
        # the first page's hard cuts
        assert "pc.z > zs * 1.03" not in blend
        assert "if (ang > 1.0472) continue;" not in blend
        assert "smoothstep(0.5, 0.95, c.a)" not in blend
        # their soft replacements
        assert "float visibility(" in blend and "smoothstep(0.015, 0.045" in blend
        assert "smoothstep(0.80, 1.0472, ang)" in blend
        assert "smoothstep(0.5, uBorder, e)" in blend
        assert "exp(-(pen[i] - pmin) / uTemp)" in blend

    def test_a_minority_colour_is_voted_down(self):
        """The down-weighting, as arithmetic.

        Two substrings proved only that the word `uConsensus` was in the file
        (review 2, "tests that don't prove their names"). The claim is that a
        sample far from the agreed colour loses weight CONTINUOUSLY and can
        never gain any, so the four lines that do that are pinned, along with
        the sign of each: the medoid weight falls off as exp(-4 D / V), the
        per-sample weight is divided by 1 + d^2, and `uConsensus` mixes between
        "no consensus" (1.0) and that -- so consensus 0 is exactly the old
        behaviour and consensus 1 never raises a weight above `w[i]`.
        """
        blend = _section(_template(), "const GLSL_SHADE", "/* ---------- main")
        assert "float cdist(" in blend
        assert "return length(a - b) / (0.08 + 0.5 * (dot(a, vec3(0.3333)) + dot(b, vec3(0.3333))));" in blend
        # the soft medoid: agreement with the other voters, weighted by theirs
        assert "float sw = v[j] * exp(-4.0 * D / max(V, 1e-6));" in blend
        # and the per-sample penalty for disagreeing with it: 1/(1+d^2), mixed
        # toward 1 by uConsensus, so it only ever multiplies the weight DOWN
        assert "float d = cdist(col[i], m0) / 0.35;" in blend
        assert "b = (w[i] + 0.05 * wmax * v[i]) * mix(1.0, 1.0 / (1.0 + d * d), uConsensus);" in blend
        # the voters are the k + 2 best, never every candidate
        assert "vthr = (n > k + 2) ? s[k + 2]" in blend

    def test_the_photometric_model_is_applied(self):
        text = _template()
        blend = _section(text, "const GLSL_SHADE", "/* ---------- main")
        assert "uGain[i] * exp(dot(uSlope[i], q) + uVig.x * r2 + uVig.y * r2 * r2)" in blend
        assert "kf.gain_slope" in text and "man.exposure.vignette" in text

    def test_every_colour_the_page_draws_goes_through_the_highlight_roll_off(self, built):
        """The roll-off, and that nothing escapes it.

        `TONE_CEILING < 0.99` says a constant is small; it does not say the
        page uses it (review 2). What makes the claim true is that EVERY path
        out of `shade` that carries a colour goes through `tone()` -- so the
        shoulder is checked, its shape is checked, and then every
        non-debug `o = vec4(...)` that carries `acc / ws` is checked to have
        `tone(` in it. The transfer is also run, in Python, against the same
        constants the shader is compiled with.
        """
        from tower.world_builder.appearance_render import build_appearance_config

        text = _template()
        blend = _section(text, "const GLSL_SHADE", "/* ---------- main")
        ceiling = float(re.search(r"#define TONE_CEILING ([0-9.]+)", blend).group(1))
        assert ceiling < 0.99                                   # the brightest output is not white
        assert "float y = uKnee + span * (1.0 - exp(-(m - uKnee) / span));" in blend
        assert "if (m <= uKnee) return x;" in blend
        knee = build_appearance_config(built.store, WORLD, SESSION)["tone_knee"]
        assert 0.3 <= knee < ceiling

        # Nothing writes the accumulated colour without the roll-off.
        writes = [ln for ln in blend.splitlines() if "acc / ws" in ln]
        assert writes, "fixture: the blend still accumulates into acc / ws"
        for line in writes:
            assert "tone(" in line, line

        # And the transfer itself: continuous at the knee, monotone, and it
        # approaches the ceiling without ever reaching it.
        import math

        def tone(m):
            if m <= knee:
                return m
            span = ceiling - knee
            return knee + span * (1.0 - math.exp(-(m - knee) / span))

        assert abs(tone(knee) - knee) < 1e-12
        last = -1.0
        for i in range(0, 401):
            m = i / 100.0                                        # past any real value
            y = tone(m)
            assert y > last, m
            last = y
        # strictly below the ceiling over everything a source can produce (a
        # value of 1.0 divided by a gain as low as 0.25), and never above it
        # for any input at all -- the shoulder only saturates ONTO the ceiling,
        # in the last bits of the float, and never past it
        assert tone(4.0) < ceiling, tone(4.0)
        for m in (10.0, 100.0, 1e6):
            assert tone(m) <= ceiling, (m, tone(m))
        assert tone(1.0) < 1.0, "a white source does not come out white"

    def test_a_changed_choice_of_sources_crossfades(self):
        text = _template()
        assert "function setChosen(" in text and "function updatePresence(" in text
        assert "uPres[i]" in _section(text, "const GLSL_SHADE", "/* ---------- main")
        maxc = int(re.search(r"const MAXC = (\d+);", text).group(1))
        cands = int(re.search(r"candidates: (\d+),", text).group(1))
        assert maxc > cands                                     # room for sources fading out

    def test_the_border_with_nothing_fades_inward_on_a_plain_background(self):
        text = _template()
        background = _section(text, "const GLSL_BACKGROUND", "`;")
        assert "texture(" not in background and "texelFetch(" not in background
        composite = _section(text, "const FS_COMPOSITE", "}`;")
        # alpha only ever shrinks from the layer's own evidence
        assert "alpha *= smoothstep(" in composite and "float alpha = layer.a;" in composite

    def test_proxy_nobody_saw_is_drawn_as_nothing_but_still_occludes(self):
        """Both halves, and the mechanism of each.

        "Still occludes" was argued from one `colorMask(...)` string appearing
        somewhere before `const P = G.blend` (review 2). What it needs is the
        whole chain: the prepass draws the WHOLE proxy with colour masked off
        and depth writing ON, the blend then runs with `LEQUAL` against it and
        depth writes OFF, the two passes share one vertex shader whose
        `gl_Position` is `invariant` (or the LEQUAL test would crack open on
        some drivers), and the unobserved fragment writes a transparent pixel
        rather than `discard`ing -- a `discard` would let what is behind it
        through, which is the whole difference between "nothing was seen here"
        and "there is nothing here".
        """
        text = _template()
        blend = _section(text, "const GLSL_SHADE", "/* ---------- main")
        # Until 2026-09-21 this wrote `vec4(0.0)` -- the background, which the
        # review measured as DARKER than the void beside it. It now writes the
        # flat haze, with `uHaze = 0` restoring the old behaviour exactly, and
        # it still writes a pixel rather than discarding.
        assert "if (uHaze <= 0.0){ o = vec4(0.0); return; }" in blend
        assert "o = vec4(haze() * uHaze, uHaze); return;" in blend
        assert "uFogLift" not in text and "uGhost" not in text
        assert "uFogLift" not in text and "uGhost" not in text
        fs_blend = _section(text, "const FS_BLEND", "const FS_FILL")
        assert "discard" not in fs_blend, "a discarded fragment would stop occluding"
        assert "invariant gl_Position;" in _section(text, "const VS_VIEW", "`;")
        pass_ = _section(text, "function blendPass(", "  let pending = false")
        prepass = pass_.index("gl.colorMask(false, false, false, false);")
        first_draw = pass_.index("drawMesh();")
        blend_prog = pass_.index("const P = G.blend;")
        assert prepass < first_draw < blend_prog, "the whole proxy writes depth before anything is shaded"
        assert "gl.depthMask(true);" in pass_[:prepass + 400]
        # and the blend then tests against it without writing it
        assert "gl.colorMask(true, true, true, true); gl.depthFunc(gl.LEQUAL); gl.depthMask(false);" in pass_

    def test_the_opening_is_the_most_drawn_frame_at_the_canvas_aspect(self):
        text = _template()
        opening = _section(text, "function openingScore(", "/* -------- context loss")
        assert "canvas.width / Math.max(1, canvas.height)" in opening
        assert "r.drawn * (0.85 + 0.15 * wide)" in opening
        assert "d * d" not in opening                           # not the old distance-squared area


class TestTheRoute:

    def test_the_header_policy_matches_the_page(self, built):
        client = _client(built)
        app = client.get(f"/worlds/{WORLD}/render", params={"viewer": V})
        assert app.status_code == 200
        assert app.headers["content-security-policy"] == APP_POLICY == _meta_policy(app.text)
        assert app.headers["cache-control"] == "no-store"
        debug = client.get(f"/worlds/{WORLD}/render", params={"transport": "tower", "viewer": V})
        assert debug.headers["content-security-policy"] == DEBUG_POLICY == _meta_policy(debug.text)

    def test_every_other_page_keeps_the_strict_policy(self, built):
        client = _client(built)
        for params in ({"representation": "surface"},
                       {"representation": "surface", "transport": "tower"}):
            r = client.get(f"/worlds/{WORLD}/render", params=params)
            assert r.status_code == 200, params
            assert r.headers["content-security-policy"] == STRICT_POLICY, params
            assert "connect-src" not in (_meta_policy(r.text) or ""), params

    def test_a_transport_outside_the_set_is_422(self, built):
        r = _client(built).get(f"/worlds/{WORLD}/render",
                               params={"transport": "https://evil.example"})
        assert r.status_code == 422

    def test_pinning_the_rung_by_query(self, built):
        client = _client(built)
        r = client.get(f"/worlds/{WORLD}/render", params={"representation": "appearance"})
        assert r.status_code == 200 and _meta(r.text, "wb-representation") == "appearance"
        rev = client.get(f"/worlds/{WORLD}/render/revision", params={"viewer": V}).json()
        assert rev["representation"] == "appearance"
        assert len(json.dumps(rev)) < 512


# ---------------------------------------------------------------------------
# review 1, B1 / M4 / M5 / m3 / m7 / m8: the page's follower, run under node
# ---------------------------------------------------------------------------


def _node():
    import shutil

    node = shutil.which("node")
    if node is None:
        pytest.skip("no node on this host to run the page's follower")
    return node


def _follower_source():
    text = _template()
    start = text.index("/* ---------- follower: what a revision poll means")
    return text[start:text.index("/* ---------- end follower */", start)]


def _run_follower(script):
    """The page's own FOLLOW unit, verbatim, then `script` (with `assert`)."""
    import subprocess

    program = ("const assert = require('assert');\n" + _follower_source()
               + "\n" + script + "\nconsole.log('follower ok');\n")
    r = subprocess.run([_node(), "-"], input=program, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "follower ok" in r.stdout, (r.stdout + r.stderr)[-3000:]


class TestTheFollower:

    def test_the_stop_transition_as_the_tower_answers_it(self, tmp_path):
        """The whole B1 sequence against the real routes: the walk's build,
        Stop (label none -> real), the final build. The page must load, keep
        its textures through the gap, keep polling, and pick the final build
        up in place (same epoch) -- never drop."""
        from fastapi.testclient import TestClient

        from tests.test_world_builder_appearance import (
            TRUSTED,
            _FakeRedactor,
            _records_carry,
        )
        from tower.world_builder import appearance as A

        w = World(tmp_path, label="none")
        _records_carry(w, _FakeRedactor())
        assert w.build(params=A.AppearanceParams.live(selection_samples=3000,
                                                       transient_detector="off"),
                       redactor_factory=_FakeRedactor).state == "ok"
        client = TestClient(_app(w.root))
        url = f"/worlds/{WORLD}/render/revision"
        q = {"session_id": SESSION, "viewer": V}
        walking = client.get(url, params=q).json()
        live_manifest = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest").json()
        w.set_label(TRUSTED)
        gap = client.get(url, params=q).json()
        _records_carry(w)
        assert w.build(params=A.AppearanceParams(selection_samples=4000, transient_detector="off"),
                       redactor_factory=_never_redact).state == "ok"
        done = client.get(url, params=q).json()
        final_manifest = client.get(f"/worlds/{WORLD}/appearance/{SESSION}/manifest").json()
        bodies = json.dumps({"walking": walking, "gap": gap, "done": done,
                             "liveMan": {"epoch": live_manifest["epoch"]},
                             "finalMan": {"epoch": final_manifest["epoch"]}})
        _run_follower("const T = " + bodies + ";\n" + STOP_SCRIPT)

    def test_withdrawn_drops_but_keeps_polling_and_a_new_epoch_replaces(self):
        _run_follower(WITHDRAWN_SCRIPT)

    def test_a_hold_is_bounded(self):
        _run_follower(HOLD_SCRIPT)

    def test_what_a_failed_poll_means(self):
        """m3: a 404 naming a world or session that is gone drops the textures;
        a transient 404 or a dropped request changes nothing."""
        _run_follower(FAILED_POLL_SCRIPT)

    def test_the_page_carries_the_actions_out(self):
        """Wiring, on the page's source: the follower decides every poll, a drop
        is not terminal, a chunk 404 refetches the manifest, a restored context
        refetches the manifest (the Tower's label check runs again), and a
        restore that never comes reloads the page."""
        text = _template()
        poll = _section(text, "async function pollOnce(", "async function follow(")
        assert "FOLLOW.decide(" in poll and "dropAppearance(" in poll and "loadRevision(" in poll
        follow = _section(text, "async function follow(", "S.refresh =")
        assert "FOLLOW.nextDelay(" in follow and "dropped" not in follow
        drop = _section(text, "function dropAppearance(", "/* -------- following")
        assert "fail(" not in drop, "a withdrawal is not a dead end"
        load = _section(text, "async function loadRevision(", "function clearTextures(")
        assert "FOLLOW.mustReplace(manifest, man)" in load and "clearTextures()" in load
        assert "e.absent && attempt === 0" in load and "continue;" in load
        assert "fetchJSON(R.manifest" in load
        restored = _section(text, 'canvas.addEventListener("webglcontextrestored"',
                            "/* -------- input")
        assert "loadRevision(" in restored
        assert "applyManifest(man" not in restored, "the old in-page manifest is never reapplied"
        lost = _section(text, 'canvas.addEventListener("webglcontextlost"',
                        'canvas.addEventListener("webglcontextrestored"')
        assert "restoreContext()" in lost and "armReloadWatchdog(RESTORE_RELOAD_MS)" in lost
        # ORDERING, which is what P-1 was about: this test passed while
        # `clearRestoreTimers()` was the FIRST statement of the restore handler,
        # disarming the reload before `buildGL()` and before the manifest and
        # every chunk were fetched again -- so the slowest, least reliable part
        # of the restore ran with no escape hatch at all, and a page could sit
        # on "Restoring..." for ever (review 2, P-1).
        assert restored.index("clearRestoreTimers();") < restored.index("armReloadWatchdog(RESTORE_FINISH_MS)"), (
            "the watchdog is re-armed, not cancelled")
        assert restored.index("armReloadWatchdog(RESTORE_FINISH_MS)") < restored.index("buildGL();"), (
            "re-armed BEFORE the work it is watching")
        # and only a restore that actually finished, one way or the other,
        # clears it
        assert restored.count("restoreSettled();") >= 3, "every exit from the restore settles it"
        watchdog = _section(text, "function armReloadWatchdog(", "canvas.addEventListener(\"webglcontextlost\"")
        assert "if (!restoring) return;" in watchdog and "location.reload();" in watchdog
        # the fetches the restore makes are themselves bounded, so a task the
        # app's scheme handler drops cannot hang the page (review 2, P-1/M-3)
        fetch = _section(text, "async function fetchBytes(", "async function fetchJSON(")
        assert "new AbortController()" in fetch and "signal: abort.signal" in fetch
        assert "setTimeout(() => abort.abort(), FETCH_TIMEOUT_MS)" in fetch


STOP_SCRIPT = r"""
const page = {revision: null, holdingSince: null, now: 0};
let r = FOLLOW.decide({ok: T.walking}, page);
assert.strictEqual(r.action, "load");
page.revision = r.revision;
assert.strictEqual(FOLLOW.decide({ok: T.walking}, page).action, "none");
r = FOLLOW.decide({ok: T.gap}, page);
assert.strictEqual(r.action, "hold", "Stop must not drop the page: " + JSON.stringify(T.gap));
assert.strictEqual(FOLLOW.nextDelay(r, 80000), FOLLOW.BASE_MS, "and it keeps asking at the base rate");
page.holdingSince = 0; page.now = 60000;
assert.strictEqual(FOLLOW.decide({ok: T.gap}, page).action, "hold");
r = FOLLOW.decide({ok: T.done}, page);
assert.strictEqual(r.action, "load");
assert.notStrictEqual(r.revision, page.revision);
assert.strictEqual(FOLLOW.mustReplace(T.liveMan, T.finalMan), false, "picked up in place, no blank");
"""

WITHDRAWN_SCRIPT = r"""
const page = {revision: "s1/appearance:b1", holdingSince: null, now: 0};
const withdrawn = {live: false, appearance: {revision: null, current: false, state: "withdrawn", epoch: null}};
let r = FOLLOW.decide({ok: withdrawn}, page);
assert.strictEqual(r.action, "drop");
assert.ok(!/Close and reopen/.test(r.reason), "not a dead end");
let delay = FOLLOW.BASE_MS;
for (let i = 0; i < 10; i++){ delay = FOLLOW.nextDelay(r, delay); }
assert.strictEqual(delay, FOLLOW.CEILING_MS, "slower, never silent");
// an old Tower that says nothing but null is treated as a withdrawal
assert.strictEqual(FOLLOW.decide({ok: {appearance: {revision: null}}}, page).action, "drop");
assert.strictEqual(FOLLOW.decide({ok: {revision: "s1/surface:1"}}, page).action, "drop");
// the rebuild comes back: load it, and a different epoch replaces what is shown
const back = {live: false, appearance: {revision: "s1/appearance:b9", current: true, state: "served", epoch: "b9"}};
assert.strictEqual(FOLLOW.decide({ok: back}, {revision: null}).action, "load");
assert.strictEqual(FOLLOW.mustReplace({epoch: "b1"}, {epoch: "b9"}), true);
assert.strictEqual(FOLLOW.mustReplace({epoch: "b1"}, {epoch: "b1"}), false);
assert.strictEqual(FOLLOW.mustReplace({}, {epoch: "b1"}), true, "an unknown epoch is never the same");
assert.strictEqual(FOLLOW.mustReplace(null, {epoch: "b1"}), false, "nothing on screen");
// A Tower-side CRASH is not a privacy event (review 2, m-15): the textures
// still go, because nothing re-checked the redaction label, but the wearer is
// not told their redaction record changed.
const broken = {live: false, appearance: {revision: null, current: false, state: "unavailable", epoch: null}};
const u = FOLLOW.decide({ok: broken}, page);
assert.strictEqual(u.action, "drop", "an unanswerable appearance still drops the textures");
assert.ok(/could not answer/.test(u.reason), "it says the Tower could not answer: " + u.reason);
assert.ok(!/redaction record/.test(u.reason), "and never blames the redaction record: " + u.reason);
assert.ok(!/Close and reopen/.test(u.reason), "not a dead end either");
assert.notStrictEqual(u.reason, FOLLOW.decide({ok: withdrawn}, page).reason);
"""

HOLD_SCRIPT = r"""
const gap = {appearance: {revision: null, state: "rebuilding"}};
assert.strictEqual(FOLLOW.decide({ok: gap}, {revision: "x", holdingSince: 0, now: FOLLOW.HOLD_MAX_MS}).action, "hold");
const r = FOLLOW.decide({ok: gap}, {revision: "x", holdingSince: 0, now: FOLLOW.HOLD_MAX_MS + 1});
assert.strictEqual(r.action, "drop");

// A BUILD THE TOWER CAN SEE IS NOT A TIMEOUT.
// The ceiling used to be a wall clock, and it measured how long the wearer
// had waited rather than whether anything was working. The photographic
// stages cost 459 s on a 385-keyframe walk and the capture guidance asks for
// two to three times that many keyframes, so a healthy build can outlast any
// fixed ceiling. While `live` is true the images stay, however long it takes.
const building = {live: true, appearance: {revision: null, state: "rebuilding"}};
assert.strictEqual(
  FOLLOW.decide({ok: building}, {revision: "x", holdingSince: 0, now: FOLLOW.HOLD_MAX_MS + 1}).action,
  "hold", "a live build holds past the ceiling");
assert.strictEqual(
  FOLLOW.decide({ok: building}, {revision: "x", holdingSince: 0, now: FOLLOW.HOLD_MAX_MS * 10}).action,
  "hold", "and keeps holding, because the ceiling is not the question");

// `live: false` is the case the ceiling is FOR: nothing is building, and
// nothing is going to.
const stalled = {live: false, appearance: {revision: null, state: "rebuilding"}};
assert.strictEqual(
  FOLLOW.decide({ok: stalled}, {revision: "x", holdingSince: 0, now: FOLLOW.HOLD_MAX_MS + 1}).action,
  "drop", "a stalled rebuild still degrades once the ceiling passes");
assert.strictEqual(
  FOLLOW.decide({ok: stalled}, {revision: "x", holdingSince: 0, now: FOLLOW.HOLD_MAX_MS}).action,
  "hold", "but not before it");
"""

FAILED_POLL_SCRIPT = r"""
const page = {revision: "s1/appearance:b1"};
assert.strictEqual(FOLLOW.decide({absent: "no world 'w1'"}, page).action, "drop");
assert.strictEqual(FOLLOW.decide({absent: "world 'w1' has no session 's1'"}, page).action, "drop");
assert.strictEqual(FOLLOW.decide({absent: "session 's1' of world 'w1' has no geometry yet"}, page).action, "none");
assert.strictEqual(FOLLOW.decide({absent: "the render revision is not served"}, page).action, "none");
assert.strictEqual(FOLLOW.decide({error: "timeout"}, page).action, "none");
assert.strictEqual(FOLLOW.nextDelay({action: "none", live: true}, 80000), FOLLOW.BASE_MS);
assert.strictEqual(FOLLOW.nextDelay({action: "none", live: null}, 10000), 20000);
"""


# ---------------------------------------------------------------------------
# fix-it nav lane: the capture envelope (NAV, run under node) and the source
# choice that draws what an oblique keyframe saw
# ---------------------------------------------------------------------------


def _nav_source():
    text = _template()
    start = text.index("/* ---------- navigation: where the camera may go")
    return text[start:text.index("/* ---------- end navigation */", start)]


# A synthetic room: a recorded walk along +x through the origin, one wall at
# z = 3 facing it, seen by every recorded camera, and nothing anywhere else.
NAV_ROOM = r"""
const up = [0, 1, 0];
const cams = [];
for (let i = 0; i <= 10; i++) cams.push([i * 0.5, 0, 0, 0, 0, 1]);
const path = NAV.makePath(cams, up);
const samples = [], off = [0], idx = [];
for (let x = -4; x <= 9; x += 0.1) for (let y = -2.5; y <= 2.5; y += 0.1){
  samples.push(x, y, 3);
  for (let k = 0; k < cams.length; k++) idx.push(k);
  off.push(idx.length);
}
const centres = Float32Array.from(cams.flatMap(c => [c[0], c[1], c[2]]));
const F = NAV.fieldJob({samples: Float32Array.from(samples), seenOff: Int32Array.from(off),
                        seenIdx: Int32Array.from(idx), centres, path});
let slices = 0;
while (!NAV.fieldWork(F, 50)) slices++;
const dir = (yaw, pitch) => [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
const V = {dir, up, fy: 1.25, aspect: 1.2};
const sup = (p, yaw, pitch = 0) => NAV.support(F, p, dir(yaw, pitch), up, V.fy, V.aspect).s;
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const at = [2.5, 0, 0];
"""


def _run_nav(script):
    """The page's own NAV unit, verbatim, a synthetic room, then `script`."""
    import subprocess

    program = ("const assert = require('assert');\n" + _nav_source() + "\n" + NAV_ROOM + "\n"
               + script + "\nconsole.log('nav ok');\n")
    r = subprocess.run([_node(), "-"], input=program, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "nav ok" in r.stdout, (r.stdout + r.stderr)[-3000:]


class TestTheEnvelope:

    def test_the_field_is_support_where_the_walk_looked_and_nothing_elsewhere(self):
        _run_nav(r"""
assert.ok(F.done && F.count > 0 && slices > 0, "built in slices, so the page stays interactive");
assert.ok(sup(at, 0) > 0.95, "facing the wall every camera saw: " + sup(at, 0));
assert.strictEqual(sup(at, Math.PI), 0, "behind the walk nothing was seen, so nothing is supported");
assert.ok(sup(at, Math.PI / 2) < 0.5, "sideways, half the view is empty");
assert.strictEqual(sup([2.5, 0, -9], 0), 0, "far outside the tube there is no support at all");
// every direction has exactly one cube cell, and opposite ones differ
const cells = new Set();
for (let i = 0; i < 4000; i++){
  const d = [Math.sin(i * 1.7) * Math.cos(i * 0.31), Math.cos(i * 1.3), Math.sin(i * 0.77)];
  const c = NAV.cellOf(d[0], d[1], d[2]);
  assert.ok(c >= 0 && c < NAV.CELLS);
  assert.notStrictEqual(c, NAV.cellOf(-d[0], -d[1], -d[2]));
  cells.add(c);
}
assert.ok(cells.size > NAV.CELLS * 0.9);
// the angle weight is the blend's tail: full to 60 degrees, none past 85
assert.strictEqual(NAV.angleWeight(1.0), 1);
assert.ok(NAV.angleWeight(1.40) > 0.4 && NAV.angleWeight(1.40) < 0.6);
assert.strictEqual(NAV.angleWeight(1.50), 0);
""")

    def test_moving_out_is_resisted_smoothly_and_never_passes_the_edge(self):
        _run_nav(r"""
for (const move of [[0, 0, -0.05], [0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.4]]){
  let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1}, maxE = 0, resisted = 0, prevStep = Infinity;
  for (let i = 0; i < 400; i++){
    const n = NAV.step(F, path, cam, {look: [0, 0], move, held: true}, 16, V);
    const stepLen = dist(n.p, cam.p);
    assert.ok(stepLen <= Math.hypot(...move) + 1e-9, "never more than was asked");
    if (NAV.pathDist(path, cam.p).e > NAV.R_SOFT / NAV.R_MAX + 0.05)
      assert.ok(stepLen <= prevStep + 1e-9, "past the soft edge each push moves less: no wall, no snap");
    prevStep = stepLen;
    maxE = Math.max(maxE, NAV.pathDist(path, n.p).e);
    resisted = Math.max(resisted, n.resisted);
    cam = n;
  }
  assert.ok(maxE <= 1 + 1e-9, "the envelope's edge is never passed: " + maxE + " " + move);
  assert.ok(maxE > 0.75, "but most of the way there is free: " + maxE + " " + move);
  assert.ok(resisted > 0.35, "and the page is told, so it can show the hint");
}
""")

    def test_a_released_camera_drifts_back_inside_without_a_jump(self):
        _run_nav(r"""
const out = [2.5, 0, -NAV.R_MAX * 0.95];
let cam = {p: out.slice(), yaw: 0, pitch: 0, floor: 1};
let e = NAV.pathDist(path, cam.p).e;
for (let i = 0; i < 600; i++){
  const n = NAV.step(F, path, cam, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
  // a fortieth of the tube in a frame: a drift, never a snap (the tube is
  // 1.5 across now, and the drift covers a proportion of what is left)
  assert.ok(dist(n.p, cam.p) < 0.02 * NAV.R_MAX, "a drift, never a snap: " + dist(n.p, cam.p));
  const e1 = NAV.pathDist(path, n.p).e;
  assert.ok(e1 <= e + 1e-9);
  e = e1; cam = n;
}
const soft = NAV.R_SOFT / NAV.R_MAX;
assert.ok(e < soft + 0.05 && e > soft - 0.05, "back to the soft edge, not dragged to the walk: " + e);
// a held camera is not moved by the drift
const held = NAV.step(F, path, {p: out.slice(), yaw: 0, pitch: 0, floor: 1}, {look: [0, 0], move: [0, 0, 0], held: true}, 16, V);
assert.deepStrictEqual(held.p, out);
""")

    def test_looking_is_free_all_the_way_round_and_the_darkness_is_told_not_enforced(self):
        """The fix-it interaction lane. An independent review measured 24.1
        degrees of reachable yaw at the opening pose and a median leftward look
        of -7.3 degrees over 16 poses: the support bound multiplied a held drag
        by about zero. A look is no longer bounded by the capture at all."""
        _run_nav(r"""
// a full turn, one 0.03 rad drag increment at a time, from the spot facing the
// only wall this synthetic room has: every step arrives whole
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1}, least = 1, dark = 0, worst = 1;
for (let i = 0; i < 210; i++){
  const n = NAV.step(F, path, cam, {look: [0.03, 0], move: [0, 0, 0], held: true}, 16, V);
  worst = Math.min(worst, (n.yaw - cam.yaw) / 0.03);
  least = Math.min(least, sup(n.p, n.yaw)); dark = Math.max(dark, n.dark);
  assert.strictEqual(n.resisted, 0, "a look is never resisted at all");
  cam = n;
}
assert.ok(Math.abs(cam.yaw - 210 * 0.03) < 1e-9, "the whole turn arrived: " + cam.yaw);
assert.ok(cam.yaw > 2 * Math.PI, "and it is more than a full circle: " + cam.yaw);
assert.ok(worst > 1 - 1e-12, "no step was ever scaled: " + worst);
assert.strictEqual(least, 0, "the turn went through the part nothing was captured from");
assert.ok(dark > 0.9, "and the page was told so, so it can SAY it: " + dark);
// the same turn the other way reaches the same places: no left/right asymmetry
let back = {p: at.slice(), yaw: 0, pitch: 0, floor: 1};
for (let i = 0; i < 210; i++) back = NAV.step(F, path, back, {look: [-0.03, 0], move: [0, 0, 0], held: true}, 16, V);
assert.ok(Math.abs(back.yaw + 210 * 0.03) < 1e-9, "and leftward is the same: " + back.yaw);
// a weak recorded pose is no longer a narrower band, because there is no band
const weak = NAV.step(F, path, {p: at.slice(), yaw: Math.PI, pitch: 0, floor: 0.05},
                      {look: [0.03, 0], move: [0, 0, 0], held: true}, 16, V);
assert.ok(Math.abs(weak.yaw - (Math.PI + 0.03)) < 1e-9, "even from a dark recorded view: " + weak.yaw);
// and standing at a dark recorded pose does not make the page quiet about the
// dark. `dark` is measured against the fixed thresholds and not against the
// floor, so the dimmest pose in the world reports the view exactly as the
// brightest one does -- the floor is there to stop a LIMIT pushing a weak
// recorded view away, and there is no limit on a look to soften.
const same = NAV.step(F, path, {p: at.slice(), yaw: Math.PI, pitch: 0, floor: 1},
                      {look: [0.03, 0], move: [0, 0, 0], held: true}, 16, V);
assert.strictEqual(weak.dark, same.dark, "the floor does not dim the warning: "
                   + weak.dark + " vs " + same.dark);
assert.ok(weak.dark > 0.9, "and it is a warning: " + weak.dark);
""")

    def test_a_recorded_view_that_is_itself_dark_is_never_pushed(self):
        _run_nav(r"""
// the wearer looked away from the wall: support 0, but it is a recorded pose
const pose = {p: at.slice(), yaw: Math.PI, pitch: 0, floor: 0.05};
const n = NAV.step(F, path, pose, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.strictEqual(n.yaw, Math.PI); assert.deepStrictEqual(n.p, at);
""")

    def test_the_cold_open_answers_the_finger_before_the_field_exists(self):
        """Review 2, item 7: at `phase: ready` the page drew the room, enabled
        its buttons and moved the camera exactly zero for 60 frames of input,
        with nothing on screen to say why. The field says where the capture
        COVERS; it is not needed to know where the camera may STAND, which is
        the recorded walk -- so the finger is answered from the first frame and
        early input still cannot leave the envelope."""
        _run_nav(r"""
const wait = NAV.step(null, path, {p: at.slice(), yaw: 0, pitch: 0},
                      {look: [2, 0.3], move: [0, 0, 0], held: true}, 16, V);
assert.strictEqual(wait.waiting, true, "it says the field is not ready");
assert.strictEqual(wait.yaw, 2, "and turns anyway, by exactly what was asked");
assert.ok(Math.abs(wait.pitch - 0.3) < 1e-9, "in both axes: " + wait.pitch);
// a move with no field is still held inside the tube around the recorded walk
let cam = {p: at.slice(), yaw: 0, pitch: 0};
for (let i = 0; i < 400; i++) cam = NAV.step(null, path, cam, {look: [0, 0], move: [0, 0.4, 0], held: true}, 16, V);
assert.ok(NAV.pathDist(path, cam.p).e <= 1 + 1e-9, "early input cannot escape: " + NAV.pathDist(path, cam.p).e);
assert.ok(cam.p[1] > 0.3, "but it did move: " + cam.p[1]);
""")
        text = _template()
        update = _section(text, "function navUpdate(", "function mulberry(")
        assert "S.waiting = !!next.waiting;" in update
        assert "if (next.waiting){" not in update, "a finger is never ignored"

    def test_the_cold_open_says_it_is_preparing_while_it_is(self):
        """The other half of review 2 item 7. Answering the finger is not the
        whole fix: a first drag now turns, and on this world it turns into the
        part of the room nobody photographed, which goes black -- and the
        sentence that explains that (`Nothing was photographed this way`) is
        exactly the one the page cannot say until the field is built. So for
        the second or so it takes, the page says it is preparing, and stops
        saying it the moment it is not."""
        text = _template()
        assert 'const PREPARING = "Preparing the view…";' in text
        build = _section(text, "  function buildNav(){", "  /* Overview:")
        assert "status(PREPARING);" in build, "it says so while the field builds"
        assert build.index("status(PREPARING);") < build.index('$("bBack").disabled = false;')
        assert 'if ($("status").textContent === PREPARING) status("");' in build, \
            "and stops saying it, without wiping a message someone else put there"

    def test_stepping_along_the_walk_glides(self):
        _run_nav(r"""
const a = {p: [0, 0, 0], yaw: 3.0, pitch: 0}, b = {p: [1, 0, 0], yaw: -3.0, pitch: 0.2};
assert.deepStrictEqual(NAV.glide(a, b, 0).p, a.p);
const end = NAV.glide(a, b, 1);
assert.ok(dist(end.p, b.p) < 1e-12 && Math.abs(Math.cos(end.yaw) - Math.cos(b.yaw)) < 1e-12);
const mid = NAV.glide(a, b, 0.5);
assert.ok(Math.abs(Math.cos(mid.yaw) - Math.cos(Math.PI)) < 1e-9, "the short way round, through pi");
let prev = a;
for (let t = 0.02; t <= 1.0001; t += 0.02){
  const g = NAV.glide(a, b, t);
  assert.ok(dist(g.p, prev.p) < 0.05, "no frame of a glide is a jump");
  prev = g;
}
// long enough to read as a flight (the Best view is a few units away and the
// old 1600 ms cap read as a teleport), and still bounded
assert.ok(NAV.glideMs(a, b) >= 400 && NAV.glideMs(a, {p: [30, 0, 0], yaw: 3, pitch: 0}) <= 2600);
assert.ok(NAV.glideMs(a, {p: [5, 0, 0], yaw: 3.0, pitch: 0}) > 1600, "a long crossing is not rushed");
// a gap in the recorded walk longer than JUMP is not a corridor
const gap = NAV.makePath([[0, 0, 0, 0, 0, 1], [NAV.JUMP + 3, 0, 0, 0, 0, 1]], up);
assert.ok(NAV.pathDist(gap, [(NAV.JUMP + 3) / 2, 0, 0]).e > 1);
""")

    def test_reachable_samples_are_inside_the_envelope_and_face_anywhere(self):
        _run_nav(r"""
let seed = 7; const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
// what a person can now reach: every direction, from every position in the tube
let lowest = 1, most = 0;
for (let i = 0; i < 80; i++){
  const r = NAV.sampleReachable(F, path, V, rand);
  assert.ok(r, "found one");
  assert.ok(NAV.pathDist(path, r.p).e <= 1, "inside the tube");
  assert.ok(Math.abs(r.pitch) <= NAV.PITCH_MAX + 1e-9, "within the neck's range");
  lowest = Math.min(lowest, r.support); most = Math.max(most, r.support);
}
assert.strictEqual(lowest, 0, "including directions nothing was captured from");
assert.ok(most > 0.9, "and directions the capture covers fully");
// the old, support-filtered set is still measurable, for comparison
for (let i = 0; i < 20; i++){
  const r = NAV.sampleReachable(F, path, V, rand, 4000, NAV.T_LO);
  assert.ok(r && r.support >= NAV.T_LO, "asked for supported views, got supported views");
}
""")


class TestTheNavigationWiring:
    """On the page's source: input goes through NAV, there is no free orbit,
    the envelope never paints, and the source choice covers first."""

    def test_every_input_goes_through_the_envelope(self):
        text = _template()
        update = _section(text, "function navUpdate(", "function mulberry(")
        assert "NAV.step(navField, navPath, cam," in update
        assert "NAV.glide(" in update, "stepping between recorded poses glides"
        frame = _section(text, "function frame(sync){", "function drawnFraction(")
        assert frame.index("navUpdate(tick)") < frame.index("currentCamera()")
        inp = _section(text, "/* -------- input ---", "/* -------- caption")
        assert "orbit" not in inp.lower(), "free orbit is gone"
        assert "feel(" in inp and "pinch" in inp.lower()
        assert 'id="bOrbit"' not in text and 'id="bOverview"' in text and 'id="hint"' in text
        assert "Not captured beyond here" in text

    def test_the_envelope_limits_the_camera_and_paints_nothing(self):
        text = _template()
        blend = _section(text, "const GLSL_SHADE", "/* ---------- main")
        composite = _section(text, "const FS_COMPOSITE", "}`;")
        for shader in (blend, composite):
            assert "navField" not in shader and "support" not in shader.lower()
        nav = _nav_source()
        for gl_call in ("gl.", "document.", "fetch(", "window."):
            assert gl_call not in nav, "NAV is pure: " + gl_call

    def test_the_source_choice_covers_what_a_source_can_see_first(self):
        text = _template()
        choose = _section(text, "function choose(", "/* -------- drawing")
        assert "sourceVisible(ready[s], z, u, v)" in choose
        assert "COVER_BONUS" in choose and "NAV.smooth(1.30, 1.48, ang)" in choose
        assert "Math.max(0, 1 - pen / maxang) * (0.5" not in choose, "the 60-degree cut is gone"
        depth = _section(text, "function renderSourceDepth(", "/* -------- loading")
        assert "gl.readPixels(0, 0, DW, DH, gl.RGBA_INTEGER, gl.UNSIGNED_INT" in depth
        blend = _section(text, "const GLSL_SHADE", "/* ---------- main")
        assert "0.08 * (1.0 - smoothstep(1.30, 1.48, ang))" in blend
        assert "(1.0 - smoothstep(1.40, 1.48, ang))" in blend


# ---------------------------------------------------------------------------
# fix-it blotch lane: cracks, voids, the capture's field of view and exposure,
# and an envelope that stops before the ugly frame
# ---------------------------------------------------------------------------


class TestTheCracksAndVoids:
    """On the page's source (the shaders run only in a browser; measured by the
    blotch lane, Glasses-scratch/wb-final-recon/fixit/blotch)."""

    def test_a_thin_crack_is_closed_only_across_one_plane_and_shaded_from_sources(self):
        text = _template()
        fill = _section(text, "const FS_FILL", "const FS_COPY")
        assert "if (dAt(p) < 1.0) discard;" in fill, "only where the proxy drew nothing"
        assert "if (!h && !v) discard;" in fill, "a gap with surface on both sides, or nothing"
        assert "if (k > uFillR) break;" in fill, "thin: bounded by the fill radius"
        assert "if (da2 >= 1.0 || db2 >= 1.0) return false;" in fill
        assert "0.03 * zb" in fill and "0.03 * za" in fill, "each side's slope predicts the other"
        assert "shade(P, normalize(N), o);" in fill, "the proxy's own shading: source pixels or nothing"
        for inpaint in ("textureLod", "uL", "texelFetch(uCol"):
            assert inpaint not in fill.split("${GLSL_SHADE}")[1], "no colour from neighbouring screen pixels"
        pass_ = _section(text, "function blendPass(", "  let pending = false")
        assert pass_.index("drawMesh();") < pass_.index("G.fill") and "lay.fb3" in pass_
        assert "Math.min(8, Math.round(OPT.fillPx * dpr))" in pass_
        # The BOUND is what the wearer is told about, so it is pinned here too:
        # a few device pixels, never more, whatever the query string says.
        assert "Math.max(0, Math.min(8, +(Q.get(\"fill\")" in text, "the radius is clamped at the source"
        caption = _section(text, "function updateCaption(", "/* -------- verification hooks")
        assert "Cracks a few pixels wide between two parts of one surface are closed" in caption, (
            "the page fills thin cracks and its caption must say so (review 2, M-5)")
        assert "nothing there is filled in" not in text, "the claim the fill made false"

    def test_a_void_is_never_brighter_than_the_room_beside_it(self):
        """The fog's ARITHMETIC, not its spelling.

        This test used to assert that the composite contained
        `coarse.rgb / max(coarse.a, 0.02)` -- the very expression that made a
        void brighter than a half-observed wall (review 2, P-2) -- so the suite
        was pinning the defect in place under a name that denied it. A
        source-level test cannot run a fragment shader, but it can check the
        two facts the result depends on, and each of them fails if the bug
        comes back:

        1. the layer the mip is built from is PREMULTIPLIED, so the coarse
           level is the mean of what is INKED (which is what
           `WORLD-BUILDER-APPEARANCE.md` section 3 already requires of any mip
           a client builds);
        2. the fog is that mean scaled by `uFog` and nothing else -- in
           particular never divided by a mean alpha, which is evidence and not
           coverage.

        The numeric claim itself is checked where it can be, by rendering:
        `Glasses-scratch/wb-final-recon/fixit/fix-ios2`.
        """
        text = _template()
        blend = _section(text, "const GLSL_SHADE", "/* ---------- main")
        composite = _section(text, "const FS_COMPOSITE", "}`;")
        # (1) premultiplied out of the shade, divided back out for this pixel
        assert "o = vec4(c * alpha, alpha);" in blend, "the layer must be premultiplied for the mip"
        assert "vec3 ink = layer.rgb / max(layer.a, 1.0 / 255.0);" in composite
        assert "mix(base, ink, alpha)" in composite, "so the drawn room is unchanged"
        # (2) the fog is uFog x the coarse mean, undivided
        assert "vec3 near = coarse.rgb;" in composite
        assert "coarse.a, 0.02" not in composite, "the unpremultiplied divide is the P-2 defect"
        assert "/ coarse.a" not in composite and "/ max(coarse" not in composite
        assert "base = mix(base, uFog * mix(vec3(lum), near, 0.1), nearby);" in composite
        # alpha only ever shrinks: the edge fade and the wide fade multiply it
        assert "float alpha = layer.a;" in composite
        assert "alpha *= smoothstep(0.5, 0.97, s / n);" in composite
        assert "alpha *= 1.0 - uWide * (1.0 - nearby);" in composite
        assert "alpha +=" not in composite and "alpha = max" not in composite
        # and "widen the fade near a big void" reads coverage, not evidence
        assert "float nearby = smoothstep(0.02, 0.35, coarse.a);" in composite
        assert "textureLod(uL, (vec2(p) + 0.5) / vec2(uSize), uFogLod)" in composite
        draw = _section(text, "function drawBlend(", "function shadeUniforms(")
        assert "Math.log2(Math.min(w, h) / 24)" in draw and "gl.generateMipmap(gl.TEXTURE_2D)" in draw

    def test_the_proxy_normals_are_read_for_obliquity(self):
        text = _template()
        assert "const nrm = hasN ? view(Int8Array, buf, off + nV * 9, nV * 3) : null;" in text
        assert "gl.vertexAttribPointer(1, 3, gl.BYTE, true, 0, 0);" in text

    def test_the_rejected_blend_changes_are_not_in_the_shader(self):
        blend = _section(_template(), "const GLSL_SHADE", "/* ---------- main")
        for rejected in ("uStretch", "uAgree", "uSlack", "planeStep", "spread", "uCoherent", "uKi"):
            assert rejected not in blend, rejected

    def test_the_matrix_inverse_the_fill_unprojects_with(self):
        text = _template()
        m = text[text.index("const M = {"):text.index("const norm3")]
        program = ("const assert = require('assert');\n" + m + r"""
const P = M.persp(1.1, 0.6, 0.05, 400), V = M.look([1, 2, 3], [0.5, 1.7, -2], [0, 1, 0]);
const A = M.mul(P, V), I = M.mul(M.inv(A), A);
for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++)
  assert.ok(Math.abs(I[c * 4 + r] - (r === c ? 1 : 0)) < 1e-4, "inverse: " + r + "," + c + " " + I[c * 4 + r]);
console.log("inv ok");
""")
        import subprocess

        r = subprocess.run([_node(), "-"], input=program, capture_output=True, text=True, timeout=60)
        assert r.returncode == 0 and "inv ok" in r.stdout, (r.stdout + r.stderr)[-2000:]


class TestTheCapturesOwnLook:

    def test_the_display_field_of_view_is_the_keyframes_plus_a_bounded_margin(self):
        """The frame is the capture's own, widened by a margin on each tangent
        (the fix-it framing lane: cut to the keyframe exactly, more than half
        of the reachable views were a clean empty wall) and capped, so no
        canvas shape and no margin can make the page look like a fisheye."""
        text = _template()
        cam = _section(text, "function currentCamera(", "function navView(")
        assert "fy: viewFovY(aspect)" in cam
        assert "const FOV_MAX_V = 1.48, FOV_MAX_H = 1.75;" in cam
        assert "const t = Math.min(OPT.viewMarginV * KT.v, OPT.viewMargin * KT.h / aspect," in cam
        assert "Math.tan(FOV_MAX_V / 2), Math.tan(FOV_MAX_H / 2) / aspect);" in cam
        assert "if (cf.fx > 0 && cf.fy > 0) KT = {v: KH / 2 / cf.fy, h: KW / 2 / cf.fx};" in text

    def test_the_display_mapping_matches_the_keyframes_brightness(self, built):
        from tower.world_builder import appearance_render as R

        config = R.build_appearance_config(built.store, WORLD, SESSION)
        assert config["exposure"] == R.DISPLAY_EXPOSURE == 1.0
        assert config["gamma"] == R.DISPLAY_GAMMA == 1.0
        assert config["tone_knee"] == R.TONE_KNEE == 0.8
        assert config["view_margin"] == R.VIEW_MARGIN == 1.4
        assert config["view_margin_v"] == R.VIEW_MARGIN_V == 1.15
        # 0.6 since the fix-it interaction lane: with the view-quality bound
        # gone from movement, the standoff is the limit a forward push actually
        # meets, and 1.0 on a world whose median scene depth is 4.7 was a fifth
        # of the room. The page's own default must agree with the served one.
        assert config["standoff"] == R.STANDOFF == 0.6
        assert 'OPT.standoff) || 0.6) * UNIT;' in _template(), "and the page's fallback is the same number"
        assert config["crack_fill_px"] == R.CRACK_FILL_PX > 0
        assert config["void_fog"] == R.VOID_FOG and config["void_wide_fade"] == R.VOID_WIDE_FADE
        text = _template()
        assert "gl.uniform1f(P.u.uExposure, OPT.exposure); gl.uniform1f(P.u.uGamma, OPT.gamma);" in text

    def test_the_caption_is_one_line_until_asked(self):
        text = _template()
        cap = _section(text, "function updateCaption(", "/* -------- verification hooks")
        assert "more.hidden = !captionOpen;" in cap and "let captionOpen = false;" in text
        assert "#caption .more[hidden]{display:none}" in text

    def test_nothing_is_drawn_before_the_opening_and_it_fades_in(self):
        text = _template()
        frame = _section(text, "function frame(sync){", "function drawnFraction(")
        assert frame.index("if (shownAt === null) return;") < frame.index("drawBlend(")
        # `finishOpening` since review 2 (M-0): a page that boots with nothing
        # placed reaches this later, when a build it can draw is served, and
        # without it that recovery would have textures and no opening pose.
        start = _section(text, "  function showFirst(){", "  async function finishOpening(")
        assert start.index("shownAt =") < start.index("frame(true);")
        # and the opening scan hands a pose back early, so the reader is not
        # looking at a black canvas for the two seconds it takes
        fin = _section(text, "async function finishOpening(", "/* -------- start ---")
        assert "opening = await chooseOpening(i => {" in fin
        # `restored ||` since the bestview lane: a reload of this tab puts the
        # camera back where it was, and the provisional is what it falls back
        # to when there is nothing stored (VISUAL-REVIEW-4 #7).
        assert "shownIndex = i; ci = i; setPose(restored || poseOf(i)); showFirst();" in fin
        assert "uShow" in _section(text, "const FS_COMPOSITE", "}`;")


class TestTheEnvelopeStopsBeforeTheUglyFrame:

    def test_the_floor_after_the_build_is_the_recorded_poses(self):
        text = _template()
        build = _section(text, "function buildNav(){", "/* Overview:")
        assert "cam.floor = floorAt(poseOf(ci));" in build and "cam.floor = floorAt(cam);" not in build

    def test_the_field_rates_quality_not_only_coverage(self):
        text = _template()
        nav = _nav_source()
        assert "* (seenQ ? seenQ[j] / 255 : 1)" in nav and "if (sampleQ) best *= sampleQ[s] / 255;" in nav
        inp = _section(text, "function* navInputSteps(", "function* voidEdgesSteps(")
        assert "NAV.smooth(0.1, 0.4, cosI)" in inp and "holeNear(X0, X1, X2) ? 0.3 : 1" in inp
        _run_nav(r"""
// the same room, but every sample seen only obliquely and beside a hole
const q = F.input;
const G2 = NAV.fieldJob(Object.assign({}, q, {seenQ: new Uint8Array(q.seenIdx.length).fill(128),
                                               sampleQ: new Uint8Array(q.samples.length / 3).fill(128)}));
while (!NAV.fieldWork(G2, 500));
const s1 = NAV.support(F, at, dir(0, 0), up, V.fy, V.aspect).s, s2 = NAV.support(G2, at, dir(0, 0), up, V.fy, V.aspect).s;
assert.ok(s1 > 0.95 && s2 < 0.3, "poor evidence is poor support: " + s1 + " " + s2);
""")

    def test_the_camera_keeps_a_distance_from_the_surface(self):
        _run_nav(r"""
// something standing 1.4 in front of the walk (the support is unchanged)
const F3 = Object.assign({}, F, {input: Object.assign({}, F.input, {samples: Float32Array.from([...F.input.samples, 2.5, 0, 1.4])})});
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1};
for (let i = 0; i < 600; i++) cam = NAV.step(F3, path, cam, {look: [0, 0], move: [0, 0, 0.02], held: true}, 16, V);
const d = NAV.nearest(F3, cam.p);
assert.ok(d >= NAV.D_MIN - 1e-6, "never nearer than D_MIN: " + d);
assert.ok(d < NAV.D_SOFT + 0.1, "but it did get close: " + d);
assert.ok(NAV.closeBound(F, [2.5, 0, 0]) === 0 && NAV.closeBound(F, [2.5, 0, 3 - NAV.D_MIN]) > 0.999);
""")

    def test_a_push_in_a_good_place_is_not_sluggish(self):
        _run_nav(r"""
// inside the free part of the band a step is taken whole
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1};
const n = NAV.step(F, path, cam, {look: [0, 0], move: [0.04, 0, 0], held: true}, 16, V);
assert.ok(Math.abs(dist(n.p, cam.p) - 0.04) < 1e-9, "a push along the walk moves what was asked: " + dist(n.p, cam.p));
""")

    def test_there_is_no_escape_hatch_left_because_there_is_nothing_to_escape(self):
        """`lookAcross` was the designed way out of the support edge: a look
        held against it glided to the next well-supported heading. Over ten
        60-frame sweeps at five poses on the canonical world it fired zero
        times (`S.crossings` stayed 0 throughout), and the only behaviour
        available at the edge was the wall. The edge is gone, so the hatch is
        gone with it, rather than being left in the page unfired."""
        _run_nav(r"""
assert.strictEqual(NAV.lookAcross, undefined, "no crossing: a look never stops");
assert.strictEqual(NAV.contentBound, undefined, "and nothing bounds a look by its content");
""")
        text = _template()
        update = _section(text, "function navUpdate(", "function mulberry(")
        for gone in ("lookAcross", "pushAcross", "S.crossings"):
            assert gone not in update, gone
        # the name survives only where the comment says why it went
        assert "function lookAcross" not in _nav_source()
        assert "NAV.lookAcross" not in text

    def test_the_walk_buttons_skip_poses_that_render_badly(self):
        _run_nav(r"""
const q = [0.95, 0.4, 0.5, 0.9, -1, 0.3];
assert.strictEqual(NAV.nextPose(q, 0, 1, 0.8).index, 3, "skips 1 and 2");
assert.strictEqual(NAV.nextPose(q, 0, 1, 0.8).skipped, 2);
assert.strictEqual(NAV.nextPose(q, 3, 1, 0.8).index, 4, "an unscored pose counts as good");
assert.strictEqual(NAV.nextPose(q, 4, 1, 0.8).index, 4, "nothing good ahead: stay");
assert.strictEqual(NAV.nextPose(q, 3, -1, 0.8).index, 0);
""")
        text = _template()
        step = _section(text, "  function step(d){", "  let poseQ")
        assert "NAV.nextPose(poseQ, from, d, POSE_MIN, poseC, POSE_CONTENT)" in step
        assert "const aim = walkTarget(ci, d);" in step, "a press covers ground first"
        assert "scorePoses();" in _section(text, "function buildNav(){", "/* Overview:")

    def test_overview_is_a_distinct_vantage(self):
        over = _section(_template(), "const OVERVIEW_AWAY", "function overview(){")
        assert "< OVERVIEW_AWAY * OPT.sceneUnit) continue;" in over
        assert "r.dist < OVERVIEW_DEPTH * ref" in over and "0.4 + 0.6 * Math.min(1, r.distance / (1.5 * ref))" in over

    def test_the_look_is_limited_by_the_neck_and_not_by_the_capture(self):
        """It used to stop at the recorded walk's own pitch range plus 0.2 rad,
        which on the canonical world gave nine of sixteen poses under 7 degrees
        of upward look. The limit is now 80 degrees each way, whatever the walk
        happened to point at, eased over its last PITCH_SOFT so it is a stop
        and not a wall."""
        _run_nav(r"""
// the recorded walk in this room looks dead level, and the look still goes up
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1}, prev = Infinity, resisted = 0;
for (let i = 0; i < 300; i++){
  const n = NAV.step(F, path, cam, {look: [0, 0.02], move: [0, 0, 0], held: true}, 16, V);
  assert.ok(n.pitch <= NAV.PITCH_MAX + 1e-9, "never past the neck: " + n.pitch);
  if (n.pitch > NAV.PITCH_MAX - NAV.PITCH_SOFT)
    assert.ok(n.pitch - cam.pitch <= prev + 1e-9, "slowing, not a wall");
  prev = n.pitch - cam.pitch; resisted = Math.max(resisted, n.resisted); cam = n;
}
assert.ok(NAV.PITCH_MAX > 1.39, "80 degrees, not 74.5: " + NAV.PITCH_MAX);
// the last of the range is eased, so it approaches the limit and never
// reaches it: 300 frames of drag get within 0.05 rad (2.7 degrees) of it
assert.ok(cam.pitch > NAV.PITCH_MAX - 0.05, "it gets there: " + cam.pitch);
assert.ok(resisted > 0.35, "and the last of it is eased: " + resisted);
for (let i = 0; i < 600; i++) cam = NAV.step(F, path, cam, {look: [0, -0.02], move: [0, 0, 0], held: true}, 16, V);
assert.ok(cam.pitch >= -NAV.PITCH_MAX - 1e-9 && cam.pitch < -NAV.PITCH_MAX + 0.05,
          "and the same downward: " + cam.pitch);
// the pitch the WALK looked at is still measured -- it is reported, not enforced
""")
        text = _template()
        assert "pitchMin" not in _section(text, "function navView(", "function supportOf(")
        assert "recordedPitch();" in text and "S.pitchLimit = [-NAV.PITCH_MAX, NAV.PITCH_MAX];" in text

    def test_pressing_overview_twice_works(self):
        text = _template()
        # the statistics used to be assigned over the verification hook
        assert "S.overviewStats = {" in text and "S.overview = {" not in text
        assert "S.overview = () => overview();" in text


# ---------------------------------------------------------------------------
# fix-it framing lane: a wider frame than the capture's, and a score that knows
# the difference between a view that is drawn and one that has something in it
# ---------------------------------------------------------------------------

# The same synthetic room, with CONTENT: the right half of the wall (x > 2.5)
# holds texture, the left half is blank. Coverage is identical across both.
NAV_CONTENT = r"""
const sampleC = new Uint8Array(samples.length / 3);
for (let s = 0; s < sampleC.length; s++) sampleC[s] = samples[s * 3] > 2.5 ? 255 : 0;
const FC = NAV.fieldJob({samples: Float32Array.from(samples), seenOff: Int32Array.from(off),
                         seenIdx: Int32Array.from(idx), centres, sampleC, path});
while (!NAV.fieldWork(FC, 50)) { /* build it */ }
const con = (p, yaw, pitch = 0) => NAV.support(FC, p, dir(yaw, pitch), up, V.fy, V.aspect).c;
const supC = (p, yaw, pitch = 0) => NAV.support(FC, p, dir(yaw, pitch), up, V.fy, V.aspect).s;
"""


class TestTheFrameAndWhatIsInIt:
    """The fix-it framing lane. The display frame is the capture's plus a
    margin; the envelope knows a drawn-but-empty view from a room view. What it
    does with that is the fix-it interaction lane's: nothing, while a person is
    looking, and a small settle for a camera the page itself placed."""

    def test_the_field_carries_content_beside_coverage(self):
        _run_nav(NAV_CONTENT + r"""
// the same wall, equally well covered from both halves
assert.ok(Math.abs(supC(at, 0.6) - supC(at, -0.6)) < 0.08, "coverage does not tell them apart");
assert.ok(con(at, 0.6) > 0.7, "the textured half: " + con(at, 0.6));
// The blank half's ABSOLUTE reading depends on how the lattice falls across
// the texture's edge (measured over this room: 0.106 at spacing 0.5, 0.233 at
// 0.6, 0.186 at 0.7, 0.242 at 0.8 -- not monotone). What the page uses is the
// separation, and it is three-to-one or better at every spacing.
assert.ok(con(at, -0.6) < 0.3, "the blank half: " + con(at, -0.6));
assert.ok(con(at, 0.6) > 3 * con(at, -0.6), "and the two are plainly different");
// richness is the 0..1 reading of it, and monotone between the thresholds
assert.strictEqual(NAV.richness(NAV.C_LO), 0);
assert.strictEqual(NAV.richness(NAV.C_HI), 1);
let last = -1;
for (let c = 0; c <= 0.5; c += 0.01){ const r = NAV.richness(c); assert.ok(r >= last - 1e-12); last = r; }
// a field built without content says nothing rather than something wrong
assert.strictEqual(NAV.support(F, at, dir(0, 0), up, V.fy, V.aspect).c, 0);
""")

    def test_a_featureless_view_does_not_slow_the_look_at_all(self):
        """The framing lane made a dull view a nudge rather than a wall (at
        worst 40% slower). The interaction lane took the nudge out too: review 2
        found that the two bounds are indistinguishable from the other end of a
        finger, and that the honest place to say "there is nothing here" is the
        picture, not the control."""
        _run_nav(NAV_CONTENT + r"""
let cam = {p: at.slice(), yaw: 0.6, pitch: 0, floor: 1}, worst = 1, steps = 0;
while (cam.yaw > -0.7 && steps < 400){
  const n = NAV.step(FC, path, cam, {look: [-0.01, 0], move: [0, 0, 0], held: true}, 16, V);
  worst = Math.min(worst, Math.abs(n.yaw - cam.yaw) / 0.01);
  cam = n; steps++;
}
assert.ok(cam.yaw <= -0.7, "the look reached the blank wall: " + cam.yaw);
assert.ok(worst > 1 - 1e-12, "and every step of the way arrived whole: " + worst);
assert.strictEqual(steps, 130, "exactly what was asked, no more and no less: " + steps);
// content still measures the room; it just does not touch the controls
assert.ok(con(at, 0.6) > 0.7 && con(at, -0.6) < 0.3);
""")

    def test_a_camera_the_page_placed_settles_and_a_look_of_the_persons_own_never_does(self):
        """The settle is the only thing that may still move the camera on its
        own, and the rule that keeps it honest is `aimed`: it is set by any
        look input and cleared only when the page places the camera. A
        deliberate look stays exactly where it was put."""
        _run_nav(NAV_CONTENT + r"""
// the PAGE placed this camera on the blank half (no `aimed`): it eases toward
// the texture
let cam = {p: at.slice(), yaw: -0.7, pitch: 0, floor: 1};
const start = cam.yaw;
for (let i = 0; i < 600; i++) cam = NAV.step(FC, path, cam, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.ok(cam.yaw > start, "it eased toward what there is to see: " + start + " -> " + cam.yaw);
assert.ok(cam.yaw - start <= NAV.C_DRIFT_MAX + 1e-6, "and no further than the budget: " + (cam.yaw - start));
// it stops of its own accord, either because the budget ran out or because the
// view now has something in it -- here, the latter
const settled = NAV.support(FC, cam.p, dir(cam.yaw, cam.pitch), up, V.fy, V.aspect);
assert.ok(NAV.richness(settled.c) >= NAV.C_SETTLE, "it settled on content: " + settled.c);
const still = NAV.step(FC, path, cam, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.strictEqual(still.yaw, cam.yaw, "and then it stays put");
// a finger down is the person's own look: it does not move at all
let held = {p: at.slice(), yaw: -0.7, pitch: 0, floor: 1};
for (let i = 0; i < 600; i++) held = NAV.step(FC, path, held, {look: [0, 0], move: [0, 0, 0], held: true}, 16, V);
assert.strictEqual(held.yaw, -0.7, "a deliberate look is never taken away");
// AND a look of the person's own, once released, is not taken away either:
// this is the one that used to move. Turn deliberately onto the blank wall,
// let go, and wait ten seconds of page time.
let aimed = {p: at.slice(), yaw: -0.4, pitch: 0, floor: 1};
for (let i = 0; i < 30; i++) aimed = NAV.step(FC, path, aimed, {look: [-0.01, 0], move: [0, 0, 0], held: true}, 16, V);
assert.strictEqual(aimed.aimed, true, "the page knows the person aimed it");
const put = aimed.yaw;
for (let i = 0; i < 600; i++) aimed = NAV.step(FC, path, aimed, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.strictEqual(aimed.yaw, put, "a deliberate look stays where it was put: " + put + " -> " + aimed.yaw);
assert.ok(NAV.richness(NAV.support(FC, aimed.p, dir(aimed.yaw, 0), up, V.fy, V.aspect).c)
          < NAV.richness(NAV.support(FC, aimed.p, dir(0.6, 0), up, V.fy, V.aspect).c) - 0.4,
          "and it is still on the blank wall, which is where it was pointed");
// and a look of the person's own gives the budget back
const after = NAV.step(FC, path, cam, {look: [-0.05, 0], move: [0, 0, 0], held: true}, 16, V);
assert.strictEqual(after.cdrift, 0);
// on a view that already has content in it there is no ease at all
let rich = {p: at.slice(), yaw: 0.6, pitch: 0, floor: 1};
for (let i = 0; i < 300; i++) rich = NAV.step(FC, path, rich, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.strictEqual(rich.yaw, 0.6, "a room view is left alone");
""")
        # the page clears `aimed` only by placing the camera itself
        text = _template()
        assert "aimed: next.aimed" in _section(text, "function navUpdate(", "function mulberry(")
        setp = _section(text, "  function setPose(pose){", "  function glideTo(")
        assert "aimed" not in setp, "placing the camera starts it un-aimed"

    def test_the_walk_buttons_skip_what_is_empty_and_cap_the_skipping(self):
        _run_nav(r"""
// drawn well enough everywhere, but 1 and 2 have nothing in them
const q = [0.95, 0.95, 0.95, 0.95], c = [0.9, 0.05, 0.05, 0.8];
assert.strictEqual(NAV.nextPose(q, 0, 1, 0.8, c, 0.28).index, 3, "an empty pose is skipped too");
// a long bad run is NOT skipped whole: the step lands on the best of it, so a
// stretch of the walk is never jumped over and the walk stays representative
const qq = [0.95], cc = [0.9];
for (let i = 0; i < 3 * NAV.MAX_SKIP; i++){ qq.push(0.5); cc.push(0.1); }
qq.push(0.95); cc.push(0.9);
qq[3] = 0.79; cc[3] = 0.27;          // the least bad of the run
const r = NAV.nextPose(qq, 0, 1, 0.8, cc, 0.28);
assert.strictEqual(r.capped, true);
assert.strictEqual(r.skipped, 2, "poses 1 and 2 were passed over, and no more");
assert.ok(r.index <= NAV.MAX_SKIP + 1, "never more than MAX_SKIP passed over: " + r.index);
assert.strictEqual(r.index, 3, "and it lands on the best of the run");
// with no content array it is the old rule exactly
assert.strictEqual(NAV.nextPose([0.95, 0.4, 0.9], 0, 1, 0.8).index, 2);
""")
        text = _template()
        cap = _section(text, "function updateCaption(", "/* -------- verification hooks")
        assert 'at most " + NAV.MAX_SKIP + " in a row' in cap, "the caption says the walk is shortened"

    def test_the_reachable_sampler_respects_the_standoff(self):
        _run_nav(r"""
// the sampler used to ignore the standoff the camera itself obeys, so the
// measured distribution held views pressed against a wall
const rand = (s => () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff)(7);
let n = 0;
for (let i = 0; i < 200; i++){
  const r = NAV.sampleReachable(F, path, V, rand);
  if (!r) continue;
  n++;
  assert.ok(NAV.nearest(F, r.p) >= NAV.D_MIN - 1e-9, "inside the standoff: " + NAV.nearest(F, r.p));
  assert.ok(NAV.pathDist(path, r.p).e <= 1 + 1e-9);
  assert.ok(r.rich >= 0 && r.rich <= 1);
}
assert.ok(n > 100, "the envelope is still reachable: " + n);
// and the grid the standoff is asked through answers exactly what the sweep did
for (let i = 0; i < 50; i++){
  const p = [rand() * 12 - 4, rand() * 5 - 2.5, rand() * 6 - 1.5];
  let b = Infinity;
  for (let j = 0; j < F.input.samples.length; j += 3){
    const dx = F.input.samples[j]-p[0], dy = F.input.samples[j+1]-p[1], dz = F.input.samples[j+2]-p[2];
    b = Math.min(b, dx*dx + dy*dy + dz*dz);
  }
  assert.ok(Math.abs(NAV.nearest(F, p) - Math.sqrt(b)) < 1e-9, "the grid is exact");
}
""")

    def test_the_scores_that_choose_a_view_all_use_the_rendered_detail(self):
        text = _template()
        assert "detail = dn ? ds / dn : 0;" in _section(text, "function drawnFraction(", "/* The opening view")
        opening = _section(text, "function openingScore(", "function chooseOpening(")
        assert "detailTerm(r.detail || 0)" in opening
        over = _section(text, "const OVERVIEW_AWAY", "function overview(){")
        assert "r.c < minContent) continue;" in over and "minContent = OVERVIEW_CONTENT" in over
        # and the content filter can never leave the button dead
        assert "while (!found.length && minContent > 0)" in over
        assert "detailTerm(r.detail || 0)" in over and "r.spread / 0.2" not in over
        poses = _section(text, "function scorePoses(", "function reset(")
        assert "poseQ[i] = r.drawn; poseC[i] = r.detail || 0;" in poses
        assert "const POSE_MIN = 0.8, POSE_CONTENT = EMPTY_DETAIL;" in text, \
            "the walk's own 'empty' cut is the calibrated one, not a second guess"

    def test_the_source_detail_comes_from_the_sources_own_pixels(self):
        """The per-sample content is measured on the keyframes themselves, on
        the same grid as their depth, and a masked pixel is never content."""
        text = _template()
        assert "const FS_DETAIL = `#version 300 es" in text
        det = _section(text, "const FS_DETAIL", "/* The blend.")
        assert "if (c.a < 0.5) continue;" in det, "a redaction box is not content"
        assert "float sd = sqrt(max(0.0, m2 / n - (m / n) * (m / n)));" in det
        src = _section(text, "async function renderSourceDepth(", "/* -------- loading and live append")
        assert "L.cpuDetail = cd;" in src and "gl.useProgram(G.detail.p);" in src
        nav = _section(text, "function* navInputSteps(", "/* Large holes in the proxy")
        assert "sourceDetail(ready[k], u, v)" in nav and "sampleC[s] = Math.round(255" in nav

    def test_the_drift_back_inside_obeys_the_standoff(self):
        """A push stopped by the standoff used to drift back to a spot INSIDE
        it, because the release drift aimed at the recorded walk and the
        recorded walk passes close to the desk."""
        text = _template()
        step = _section(text, "  function step(F, path, cam, ctl, dt, V){", "  /* A uniformly random reachable view")
        assert "if (nearest(F, to) >= Math.min(D_MIN, nearest(F, out.p))) out.p = to;" in step
        _run_nav(r"""
// a surface right on the walk, closer than the standoff
const near = Float32Array.from([...F.input.samples, 2.5, 0, 0.4]);
const F4 = Object.assign({}, F, {input: Object.assign({}, F.input, {samples: near})});
let cam = {p: [2.5, 0, -0.9], yaw: 0, pitch: 0, floor: 1};
const d0 = NAV.nearest(F4, cam.p);
for (let i = 0; i < 400; i++) cam = NAV.step(F4, path, cam, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
const d1 = NAV.nearest(F4, cam.p);
assert.ok(d1 >= Math.min(NAV.D_MIN, d0) - 1e-9, "the drift never went inside the standoff: " + d0 + " -> " + d1);
""")

    def test_the_edge_hint_is_for_a_push_and_the_dark_one_is_only_an_explanation(self):
        """Two different things, and they must not be confused. The edge hint
        means the page stopped you; the dark hint means the page did NOT, and
        the room ahead is dark because nobody photographed it."""
        text = _template()
        update = _section(text, "function navUpdate(", "function mulberry(")
        assert "hint(next.hard || 0, HINT_EDGE);" in update
        # The dark one is no longer said HERE at all. It was `hint(0.45,
        # HINT_DARK)` throttled to one showing per 4,000 ms, and over a
        # 24-step turn from the opening the last review saw it at three steps
        # and at NONE of the six that render a 100% black frame. It is now a
        # state the page HOLDS for as long as the view is on nothing
        # (`updateDark`), so the deepest black is the best explained.
        assert "hint(0.45, HINT_DARK);" not in update
        assert "if (looking && next.dark > 0.9) S.darkLooks" in update
        assert 'const HINT_EDGE = "Not captured beyond here";' in text
        assert 'const HINT_DARK = "Nothing was photographed this way";' in text
        step = _section(text, "  function step(F, path, cam, ctl, dt, V){", "  /* A uniformly random reachable view")
        # the edge the hint is for is the tube and the standoff, and nothing else
        assert "const bAt = p => Math.max(posB(p), ready ? closeBound(F, p) : 0);" in step
        # and the dark one is about whether the view is on ANYTHING, not about
        # whether it is a good view: `lookBound` is the QUALITY band (T_LO, the
        # support at which three views in four render well), and asking it here
        # made `dark` exactly 1 at eight of nine sampled recorded poses on the
        # canonical world, including ones the page draws 99.4% of -- so thirty
        # frames of drag at the opening raised "Nothing was photographed this
        # way" five times, over a photograph of the room (2026-09-20, fix-it
        # orient lane; ORIENT.md §2.3).
        assert "out.dark = darkness(" in step
        assert "lookBound" not in step, "the quality band is not the darkness band"
        _run_nav(r"""
// the two bands, and the gap between them that the bug lived in
assert.ok(NAV.DARK_HI < NAV.T_LO / 2, "darkness is asked far below the quality band");
assert.strictEqual(NAV.darkness(0), 1);
assert.strictEqual(NAV.darkness(NAV.DARK_HI), 0);
assert.strictEqual(NAV.lookBound(0.65, 1), 1, "0.65 support is a poor view...");
assert.strictEqual(NAV.darkness(0.65), 0, "...and it is not a view of nothing");
let prev = -1;
for (let s = 0; s <= 1; s += 0.01){
  const d = NAV.darkness(s);
  assert.ok(d >= 0 && d <= 1 && (prev < 0 || d <= prev + 1e-12));
  prev = d;
}
""")
        _run_nav(r"""
// a look never raises `resisted`, so the edge hint can never fire for one
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1};
for (let i = 0; i < 300; i++){
  cam = NAV.step(F, path, cam, {look: [0.02, 0], move: [0, 0, 0], held: true}, 16, V);
  assert.strictEqual(cam.resisted, 0);
  assert.strictEqual(cam.hard, 0);
}
// and a push against the tube does raise it
let out = {p: at.slice(), yaw: 0, pitch: 0, floor: 1}, hard = 0;
for (let i = 0; i < 400; i++){
  out = NAV.step(F, path, out, {look: [0, 0], move: [0, 0.05, 0], held: true}, 16, V);
  hard = Math.max(hard, out.hard);
}
assert.ok(hard > 0.35, "the edge hint fires for the edge: " + hard);
""")


# ---------------------------------------------------------------------------
# fix-it interaction lane: what a finger can actually reach
# ---------------------------------------------------------------------------


class TestWhatAFingerCanReach:
    """Review 2 measured the page through its own input path and found a
    24.1-degree slot: looking was bounded by the capture, moving was bounded by
    the capture FOUR times over (the tube, the standoff, the support of the
    view from where you were going, and its content). Looking is now free and
    moving is bounded by the two limits that are true."""

    def test_a_deliberate_push_travels_what_the_envelope_really_allows(self):
        _run_nav(r"""
// 40 frames of 0.112, the review's own push: 4.48 units asked, along the walk
let cam = {p: [0.2, 0, 0], yaw: Math.PI / 2, pitch: 0, floor: 1};
const a = cam.p.slice();
for (let i = 0; i < 40; i++) cam = NAV.step(F, path, cam, {look: [0, 0], move: [0, 0, 0.112], held: true}, 16, V);
const alongWalk = dist(cam.p, a);
assert.ok(alongWalk > 4.0, "a push along the walk arrives: " + alongWalk + " of 4.48");
// sideways it is the tube that stops it, and the tube is now 1.5 across
let side = {p: [2.5, 0, 0], yaw: 0, pitch: 0, floor: 1};
for (let i = 0; i < 40; i++) side = NAV.step(F, path, side, {look: [0, 0], move: [0.112, 0, 0], held: true}, 16, V);
const across = Math.abs(side.p[2] - 0) + Math.hypot(side.p[0] - 2.5, side.p[1]);
assert.ok(NAV.pathDist(path, side.p).e <= 1 + 1e-9, "never out of the tube");
assert.ok(NAV.R_MAX >= 1.5 && NAV.R_SOFT >= 0.7,
          "and the tube is wide, and free over most of itself: " + NAV.R_MAX + " " + NAV.R_SOFT);
// where the view is going is NOT a limit: this push ends facing away from the
// only wall the room has, and it still travels
let away = {p: [2.5, 0, 0], yaw: Math.PI, pitch: 0, floor: 1};
const b = away.p.slice();
for (let i = 0; i < 40; i++) away = NAV.step(F, path, away, {look: [0, 0], move: [0, 0, 0.112], held: true}, 16, V);
assert.ok(dist(away.p, b) > 0.9, "into the dark is still a move: " + dist(away.p, b));
assert.strictEqual(NAV.support(F, away.p, dir(away.yaw, 0), up, V.fy, V.aspect).s, 0,
                   "and it really is dark there");
""")

    def test_the_button_does_not_promise_an_overview_this_capture_cannot_give(self):
        """What was wrong with this button was its name and how fast it
        arrived, not where it went. The destination is the framing lane's
        measured choice and is deliberately unchanged: re-weighting it for
        landscape was tried in this lane and made the view worse (88.1 % drawn
        against 99.8 %, and a shredded tear where a chair used to be)."""
        text = _template()
        assert ">Best view</button>" in text and ">Overview</button>" not in text
        cap = _section(text, "function updateCaption(", "/* -------- verification hooks")
        assert "Turning is free" in cap and "Moving is not free" in cap
        assert "never stood back from the desk" in cap
        over = _section(text, "const OVERVIEW_AWAY", "function overview(){")
        assert "* (0.5 + 0.5 * Math.min(1, (r.depthSpread || 0) / (OVERVIEW_RANGE * ref)))" in over
        score = over[over.index("f.rendered ="):]
        assert "aspect" not in score[:score.index(";")], \
            "the score does not depend on the orientation"
        assert "TRIED AND REJECTED" in over, "and the rejected experiment is recorded, not repeated"
        # the flight: a five-unit crossing is no longer clipped to 1600 ms
        _run_nav(r"""
const from = {p: [-2.33, 1.64, -4.00], yaw: 0.796, pitch: -0.164};
const to   = {p: [-1.31, -0.35, 0.82], yaw: 1.047, pitch: -0.500};
const ms = NAV.glideMs(from, to);
assert.ok(ms > 2000 && ms <= 2600, "the Best view is flown, not jumped: " + ms + " ms");
""")

    def test_the_about_text_is_legible_over_a_bright_frame(self):
        """Review 2, item 9: the caption's gradient scrim stops short of the
        expanded text, and over the cream door the tail of the paragraph that
        keeps the page honest was grey on cream."""
        text = _template()
        style = _section(text, "#caption{position:fixed", "#status{position:fixed")
        assert "background:rgba(8,10,13,.88)" in style, "the About text carries its own plate"
        assert "text-shadow" in style
        # and so does the status line, which now speaks over the opening frame
        # rather than only over a black loading screen
        st = _section(text, "#status{position:fixed", "#msg{position:fixed")
        assert "background:rgba(8,10,13,.82)" in st and "text-shadow" in st
        assert "#status:empty{display:none}" in st, "and shows nothing when it has nothing to say"


# ---------------------------------------------------------------------------
# fix-it orient lane: the orientation cue, the confidence fade, the Best view's
# re-swept score, and the cold-open window
# ---------------------------------------------------------------------------


class TestKnowingWhichWayTheRoomIs:
    """A look is free now, and on this capture more than half of a full turn
    was never photographed: five consecutive 30-degree steps of pure black
    (INTERACTION.md §3.1). Truthful, and with nothing on screen it reads as a
    crash. The page carries a compass of COVERAGE and a way back."""

    def test_the_ring_is_the_capture_seen_from_where_you_stand(self):
        _run_nav(r"""
const prof = NAV.ringProfile(F, at, V, 36);
assert.strictEqual(prof.length, 36);
// the synthetic room is one wall at z = 3 that every recorded camera saw
const facing = Math.round(0 / (2 * Math.PI / 36));
assert.ok(prof[facing] > 0.95, "the bin that faces the wall is covered: " + prof[facing]);
assert.strictEqual(prof[18], 0, "the bin that faces away from it is not");
// and it is the SAME quantity the rest of the page calls support, at the same
// field of view -- it reports coverage, never content
for (const i of [0, 4, 9, 18, 27]){
  const s = NAV.support(F, at, dir(i * 2 * Math.PI / 36, 0), up, V.fy, V.aspect).s;
  assert.ok(Math.abs(prof[i] - s) < 1e-6, "bin " + i + ": " + prof[i] + " vs " + s);
}
// nowhere near the capture there is nothing to report, and nothing invented
const far = NAV.ringProfile(F, [2.5, 0, -9], V, 36);
assert.ok(Array.from(far).every(s => s === 0));
""")

    def test_the_way_back_is_the_nearest_covered_heading(self):
        _run_nav(r"""
const prof = NAV.ringProfile(F, at, V, 36);
// standing with your back to the only wall in the world
const b = NAV.bestHeading(prof, Math.PI);
assert.ok(b, "there is a way back and the page can say so");
assert.ok(Math.abs(b.delta) > 2.5, "and it turns you most of the way round: " + b.delta);
assert.ok(b.support > 0.9, "to a heading that really is covered: " + b.support);
// the yaw is continuous with the one given, so a glide turns the SHORT way
assert.ok(Math.abs(b.yaw - Math.PI) <= Math.PI + 1e-9);
assert.ok(Math.abs(Math.cos(b.yaw) - 1) < 0.1, "and it faces the wall: " + b.yaw);
// facing the wall already, it barely asks you to move
const b0 = NAV.bestHeading(prof, 0);
assert.ok(Math.abs(b0.delta) < 0.2, "already facing it: " + b0.delta);

// NEAREST, not best: a slightly worse heading at your elbow beats a slightly
// better one behind you, and a much better one behind you still wins.
const n = 36, mk = (i, v) => { const p = new Float32Array(n); p[i] = v; return p; };
const near = mk(1, 0.70), farBin = mk(18, 0.80);
const both = new Float32Array(n); both[1] = 0.70; both[18] = 0.80;
assert.strictEqual(NAV.bestHeading(both, 0).delta, NAV.bestHeading(near, 0).delta,
                   "0.80 half a turn away does not beat 0.70 at your elbow");
const both2 = new Float32Array(n); both2[1] = 0.70; both2[18] = 0.99;
assert.strictEqual(NAV.bestHeading(both2, 0).delta, NAV.bestHeading(farBin, 0).delta,
                   "0.99 half a turn away does");
assert.ok(NAV.RING_TURN_PENALTY > 0 && NAV.RING_TURN_PENALTY < 1);
// and where NOTHING is covered there is no way back, and the page says nothing
assert.strictEqual(NAV.bestHeading(new Float32Array(n), 0), null);
assert.strictEqual(NAV.bestHeading(mk(3, NAV.RING_FLOOR * 0.5), 0), null);
""")

    def test_a_camera_the_page_placed_eases_back_toward_the_capture(self):
        """The settle used to ask only for CONTENT, and where the glasses
        never looked there is no content anywhere near, so its gradient was
        exactly zero and a page-placed camera on nothing sat on nothing. It
        now falls back to where the capture IS. It is a nudge at the edge and
        not a way home -- the whole budget is C_DRIFT_MAX -- and it still
        never touches a look the person made."""
        _run_nav(r"""
const away = 1.6;                                  // sideways: covered one way, not the other
const run = (cam0, held) => {
  let cam = Object.assign({p: at.slice(), pitch: 0, floor: 1, cdrift: 0}, cam0);
  for (let i = 0; i < 400; i++)
    cam = NAV.step(F, path, cam, {look: [0, 0], move: [0, 0, 0], held}, 16, V);
  return cam;
};
const placed = run({yaw: away, aimed: false}, false);
assert.ok(placed.yaw < away - 0.05, "a camera the page placed eases toward the capture: "
          + away + " -> " + placed.yaw);
assert.ok(away - placed.yaw <= NAV.C_DRIFT_MAX + 1e-6, "and no further than the budget: "
          + (away - placed.yaw));
assert.ok(sup(at, placed.yaw) > sup(at, away), "toward, not away");
const aimed = run({yaw: away, aimed: true}, false);
assert.strictEqual(aimed.yaw, away, "a look of the person's own is never moved");
const held = run({yaw: away, aimed: false}, true);
assert.strictEqual(held.yaw, away, "and nothing drifts while a finger is down");
// the wide probe is a fallback, not a replacement: the content gradient is
// still asked first
assert.ok(NAV.C_GRAD_WIDE > NAV.C_GRAD_EPS);
""")

    def test_the_page_carries_the_cue_and_the_control(self):
        text = _template()
        assert 'id="compass"' in text and 'id="back"' in text
        assert ">Face the room</button>" in text
        # it is a compass of coverage, and the page says so where a person reads
        cap = _section(text, "function updateCaption(", "/* -------- verification hooks")
        assert "compass of what was photographed" in cap
        assert "not that there is anything worth seeing" in cap
        assert "turns you \u2014 without moving you \u2014 to the nearest" in cap
        cue = _section(text, "/* -------- the orientation ring", "const KEY_SPEED")
        # the ring is recomputed only when the camera has MOVED, and the profile
        # comes from the page's own support, not from anything invented
        assert "NAV.ringProfile(navField, at, navView(), RING_BINS)" in cue
        assert "RING_MOVE" in cue
        # and everything the cue reads is the pose that is DRAWN, so a
        # verification hook pinning an exact camera does not leave the ring
        # pointing at wherever `cam` happens to be
        assert "function shownPose()" in cue
        for line in ("const at = shownPose().p;", "const yaw = shownPose().yaw;",
                     "const q = shownPose();", "const here = shownPose();",
                     "NAV.bestHeading(ringProf, shownPose().yaw)"):
            assert line in cue, line
        # the cue is drawn in world yaw with the sense DERIVED, not assumed
        assert "function yawSign()" in cue and "-Math.PI / 2 + sgn * (i * step - yaw)" in cue
        # the way back turns and never translates
        face = cue[cue.index("function faceTheRoom("):]
        assert "glideTo({p: here.p.slice(), yaw: b.yaw, pitch: bestPitch}, null)" in face
        # a glide, so a real finger cancels it: `pointerdown` calls `interrupt`
        assert "if (glideState) interrupt();" in _section(text, 'canvas.addEventListener("pointerdown"',
                                                          'canvas.addEventListener("pointermove"')
        # the edge arrow appears only on a view that really is on nothing, with
        # a gap between the two thresholds so it cannot blink on the boundary
        assert "BACK_ON = 0.60, BACK_OFF = 0.25" in cue
        assert "backOn ? dark > BACK_OFF : dark > BACK_ON" in cue
        # and it is drawn, not typed: a glyph could render as a box
        style = _section(text, "#back{position:fixed", "#back.on{")
        assert "border-left:17px solid currentColor" in style
        back = cue[cue.index("function updateBack("):cue.index("/* THE DARK STATE")]
        assert "textContent" not in back

    def test_the_ring_and_the_control_are_the_same_control(self):
        text = _template()
        wiring = _section(text, '$("bOverview").onclick', '$("bReset").onclick')
        for el in ("bBack", "compass", "back"):
            assert f'$("{el}").onclick = () => faceTheRoom();' in wiring, el


class TestTheConfidenceFade:
    """The proxy carries one byte a vertex saying how well that vertex was
    measured (the geometry-confidence lane). Where it is low the blend tears
    real photographs over wrong geometry, and the honest thing to draw is what
    the page already draws where it knows nothing."""

    def test_the_band_is_a_band_and_it_is_the_phone_level_s(self, built):
        from tower.world_builder import appearance_render as R

        assert 0 < R.CONFIDENCE_LO < R.CONFIDENCE_HI < 1
        # the phone level, not the archive level: the validated level-0 band
        # (0.30-0.50) fades 19% of the opening view on the decimated proxy
        # the lane validated 0.30-0.50 on the ARCHIVE level; on the decimated
        # proxy the fade has to be finished well before that band even starts
        assert R.CONFIDENCE_HI < 0.40, "a level-0 threshold here would delete the room"
        config = R.build_appearance_config(built.store, WORLD, SESSION)
        assert config["confidence_lo"] == R.CONFIDENCE_LO
        assert config["confidence_hi"] == R.CONFIDENCE_HI
        # and the page's own fallback agrees with what is served
        text = _template()
        assert f"CONFIG.confidence_lo ?? ({int(round(R.CONFIDENCE_LO * 255))} / 255)" in text
        assert f"CONFIG.confidence_hi ?? ({int(round(R.CONFIDENCE_HI * 255))} / 255)" in text

    def test_it_fades_to_the_void_and_never_to_transparent(self):
        shade = _section(_template(), "const GLSL_SHADE", "const FS_BLEND")
        assert "float k = smoothstep" not in shade, "one rule, one place"
        assert "smoothstep(uConfLo, uConfHi, gConf) : 1.0" in shade, "a band, never a cut"
        assert "c = mix(uHaze > 0.0 ? haze() : background(), c, confKeep());" in shade, \
            "toward the colour the page already uses for unobserved space"
        # the EVIDENCE is untouched: alpha is what says anyone saw the place,
        # the depth still hides what is behind, and no pixel goes see-through
        assert "o = vec4(c * alpha, alpha);" in shade
        for bad in ("alpha *= confKeep", "alpha = alpha * confKeep", "alpha *= k"):
            assert bad not in shade, bad
        # unknown is not zero
        assert "uConfOn > 0.5" in shade
        assert "float gConf = 1.0;" in shade, "1 where nothing is known"

    def test_a_proxy_without_the_channel_is_drawn_at_full_strength(self):
        text = _template()
        apply = _section(text, "async function applyManifest(", "  let loadGeneration")
        assert "cfd.present === true" in apply, "present: false is UNKNOWN, not zero"
        upload = _section(text, "  function uploadMesh(){", "  function meshUniforms(")
        assert "gl.disableVertexAttribArray(2); gl.vertexAttrib1f(2, 1);" in upload
        # the channel rides the colour block already on the wire: no copy, no bytes
        assert "gl.vertexAttribPointer(2, 1, gl.UNSIGNED_BYTE, true, 3, confChannel)" in upload
        shade = _section(text, "  function shadeUniforms(", "  function blendPass(")
        assert "confChannel !== null && OPT.confHi > OPT.confLo" in shade

    def test_the_attribute_slots_are_fixed_so_three_programs_agree(self):
        vs = _section(_template(), "const VS_VIEW = ", "/* What \"not captured\" looks like")
        assert "layout(location = 0) in vec3 aQ;" in vs
        assert "layout(location = 1) in vec3 aN;" in vs
        assert "layout(location = 2) in float aConf;" in vs

    def test_what_the_fade_takes_can_be_measured(self):
        text = _template()
        assert "if (uMode == 11){ o = vec4(alpha * (1.0 - confKeep()), alpha, gConf, 1.0); return; }" \
            in text, "the removed area and the drawn area, so the share is one ratio"
        assert "S.fadeCost = () => {" in text
        assert "ofDrawn:" in text and "ofFrame:" in text


class TestTheBestViewWasReSweptAgainstTheWiderTube:
    """The interaction lane widened the tube from 1.0 to 1.5 and the candidate
    set went from 205 to 406. Nobody chose what that did to the Best view: the
    winner moved half a unit back, from 99.8% drawn to 88.7% with a shredded
    corner. The score could not see the difference, because everything it
    measured -- drawn, deep, detailed, with range -- is as true of a photograph
    smeared over wrong geometry as of a photograph of the room. The confidence
    channel can see it, and the page now draws it, so the score reads it off
    the same render."""

    def test_the_score_will_not_recommend_geometry_the_fade_eats(self):
        over = _section(_template(), "const OVERVIEW_AWAY", "function overview(){")
        # the bestview lane put `* cleanTerm(r.torn)` after it, so the
        # confidence term is no longer the last factor on the line
        assert "* soundTerm(r.faded)\n" in over
        assert "OVERVIEW_FADED_REF" in over and "OVERVIEW_SOUND_FLOOR" in over
        # the term itself, run: bounded, monotone, and flat once a frame is wrecked
        src = over[over.index("const OVERVIEW_FADED_REF"):over.index("function findOverview")]
        _run_plain(src + r"""
assert.strictEqual(soundTerm(0), 1);
assert.ok(Math.abs(soundTerm(OVERVIEW_FADED_REF) - OVERVIEW_SOUND_FLOOR) < 1e-12);
assert.ok(Math.abs(soundTerm(1) - OVERVIEW_SOUND_FLOOR) < 1e-12, "and it saturates, never below");
let prev = 2;
for (let f = 0; f <= 0.2; f += 0.002){
  const s = soundTerm(f);
  assert.ok(s <= prev + 1e-12 && s >= OVERVIEW_SOUND_FLOOR - 1e-12);
  prev = s;
}
""")

    def test_on_the_measured_candidates_it_flips_the_winner_back(self):
        """The two decisive rows of the 12-candidate table this lane measured
        on the canonical world (ORIENT.md §4). c03 is the vantage the wider
        tube handed the old score -- pulled back, 88.7% drawn, and 3.07% of
        what it draws is geometry the fade removes. c11 is the vantage the
        score chose on the 1.0-unit tube, 99.8% drawn and 0.27% faded. Without
        the confidence term c03 wins by 1.4%; with it c11 wins by 4.6%."""
        over = _section(_template(), "const OVERVIEW_AWAY", "function overview(){")
        src = over[over.index("const OVERVIEW_FADED_REF"):over.index("function findOverview")]
        _run_plain(src + r"""
const ref = 4.106, DETAIL_REF = 0.12;
const dterm = d => Math.max(0, Math.min(1, d / DETAIL_REF));
const base = t => t.drawn * (0.4 + 0.6 * Math.min(1, t.distance / (1.5 * ref)))
  * (0.5 + 0.5 * Math.max(0, t.toward)) * (0.15 + 0.85 * dterm(t.detail))
  * (0.5 + 0.5 * Math.min(1, t.range / ref));
const c03 = {drawn: 0.887, detail: 0.1617, range: 4.92, toward: 0.941, distance: 6.933, faded: 0.0307};
const c11 = {drawn: 0.998, detail: 0.1878, range: 3.69, toward: 0.891, distance: 5.617, faded: 0.0027};
assert.ok(base(c03) > base(c11), "the old score prefers the wrecked frame");
assert.ok(base(c11) * soundTerm(c11.faded) > base(c03) * soundTerm(c03.faded),
          "and the confidence term puts it back");
""")


class TestTheColdOpenWindow:
    """`Preparing the view…` is how long the page cannot yet say why a
    direction is dark. Widening the tube nearly doubled it, because the field
    it waits on nearly doubled. A coarse field answers the same question."""

    def test_the_field_is_usable_before_it_is_exact(self):
        build = _section(_template(), "  function buildNav(){", "  /* Overview:")
        assert "startJob(NAV.SPACING * NAV_COARSE);" in build, "coarse first"
        assert "startJob(NAV.SPACING);" in build, "then the real one"
        assert build.index("startJob(NAV.SPACING * NAV_COARSE);") < build.index("startJob(NAV.SPACING);")
        # what the coarse field is enough for, and what it is not
        assert '$("bBack").disabled = false;' in build
        assert '$("bOverview").disabled = coarse;' in build
        assert "scorePoses();" in build
        over = _section(_template(), "  function overview(){", "  /* -------- source choice")
        assert "!navFine" in over, "and the hook refuses too, not only the button"

    def test_a_coarser_field_answers_the_same_question_for_an_eighth(self):
        _run_nav(r"""
const C = NAV.fieldJob(F.input, NAV.SPACING * 2);
let slicesC = 0;
while (!NAV.fieldWork(C, 50)) slicesC++;
assert.strictEqual(C.spacing, NAV.SPACING * 2);
assert.strictEqual(F.spacing, NAV.SPACING);
// a third here, where the synthetic tube is short enough for its ends to
// dominate; 866 against 3,851 on the canonical world
assert.ok(C.count * 3 < F.count, "far fewer voxels: " + C.count + " of " + F.count);
const supC = (p, yaw) => NAV.support(C, p, dir(yaw, 0), up, V.fy, V.aspect).s;
let worst = 0;
for (let i = 0; i < 24; i++){
  const yaw = i * Math.PI / 12;
  worst = Math.max(worst, Math.abs(supC(at, yaw) - sup(at, yaw)));
  // and the two agree about which way the capture IS, which is all the cue
  // and the dark hint ask of it
  assert.strictEqual(supC(at, yaw) > 0.5, sup(at, yaw) > 0.5, "yaw " + yaw);
}
assert.ok(worst < 0.2, "and nowhere far apart: " + worst);
assert.strictEqual(supC([2.5, 0, -9], 0), 0, "outside the tube it is still nothing");
""")


def _run_plain(script):
    """A fragment of the page, lifted and run under node with `assert`."""
    import subprocess

    program = "const assert = require('assert');\n" + script + "\nconsole.log('plain ok');\n"
    r = subprocess.run([_node(), "-"], input=program, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "plain ok" in r.stdout, (r.stdout + r.stderr)[-3000:]


def _encoding_source():
    text = _template()
    start = text.index("/* ---------- encoding: which texture set this phone gets")
    return text[start:text.index("/* ---------- end encoding */", start)]


def _run_encoding(script):
    """The page's own ENC unit, verbatim, then `script` (with `assert`)."""
    import subprocess

    program = ("const assert = require('assert');\n" + _encoding_source()
               + "\n" + script + "\nconsole.log('enc ok');\n")
    r = subprocess.run([_node(), "-"], input=program, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and "enc ok" in r.stdout, (r.stdout + r.stderr)[-3000:]


ENC_ROOM = r"""
const astcChunk = {encoding: "astc-6x6-rgba", tier: "phone", digest: "a"};
const webpChunk = {encoding: "webp-rgba", tier: "phone", digest: "b"};
const kf = (id, rank, chunks, tier) => ({id, rank, tier: tier || "phone", chunks});
const both = {encodings: {"astc-6x6-rgba": {available: true}, "webp-rgba": {available: true}},
              chunks: [astcChunk, webpChunk],
              selection: {phone_budget: 128},
              keyframes: [kf("k3", 3, {"astc-6x6-rgba": {digest: "a", slot: 0}, "webp-rgba": {digest: "b", slot: 0}}),
                          kf("k1", 1, {"astc-6x6-rgba": {digest: "a", slot: 1}, "webp-rgba": {digest: "b", slot: 1}}),
                          kf("k2", 2, {"webp-rgba": {digest: "b", slot: 2}}),
                          kf("t9", 0, {"astc-6x6-rgba": {digest: "a", slot: 9}}, "tower")]};
const noAstcBuild = {encodings: {"webp-rgba": {available: true}}, chunks: [webpChunk],
                     selection: {phone_budget: 128}, keyframes: both.keyframes};
const astcDeclaredButUnbuilt = {encodings: {"astc-6x6-rgba": {available: true}},
                                chunks: [webpChunk], selection: {phone_budget: 128},
                                keyframes: both.keyframes};
"""


class TestTheEncodingChoice:
    """`ENC`, the page's own unit, under node.

    Pulled out of `main()` by review 2: the ASTC fallback, the phone tier and
    the layer budget are three things only a device could check, and there was
    no unit for any of them -- a grep for `astc`, `capacity` or `maxLayers` in
    this file returned nothing but two context-loss `_section` calls.
    """

    def test_astc_needs_the_device_the_manifest_and_an_actual_chunk(self):
        _run_encoding(ENC_ROOM + r"""
assert.strictEqual(ENC.choose(both, {astc: true}), "astc-6x6-rgba");
assert.strictEqual(ENC.choose(both, {astc: false}), "webp-rgba", "no extension on this device");
assert.strictEqual(ENC.choose(both, {astc: true, forceWebp: true}), "webp-rgba", "asked for WebP");
assert.strictEqual(ENC.choose(noAstcBuild, {astc: true}), "webp-rgba", "the Tower built none");
assert.strictEqual(ENC.choose(astcDeclaredButUnbuilt, {astc: true}), "webp-rgba",
                   "declared but no phone chunk to fetch");
assert.strictEqual(ENC.choose({}, {astc: true}), "webp-rgba", "an empty manifest is not ASTC");
assert.strictEqual(ENC.choose(both, {}), "webp-rgba", "a device that did not say has no extension");
""")

    def test_the_phone_tier_is_ranked_and_the_tower_tier_is_never_drawn(self):
        _run_encoding(ENC_ROOM + r"""
const astc = ENC.phoneKeyframes(both, "astc-6x6-rgba").map(k => k.id);
assert.deepStrictEqual(astc, ["k1", "k3"], "rank order, phone tier, and this encoding only");
const webp = ENC.phoneKeyframes(both, "webp-rgba").map(k => k.id);
assert.deepStrictEqual(webp, ["k1", "k2", "k3"], "the WebP set is a superset here");
assert.ok(!webp.includes("t9") && !astc.includes("t9"), "the Tower tier is never drawn on a phone");
assert.deepStrictEqual(ENC.phoneKeyframes({}, "webp-rgba"), []);
""")

    def test_the_layer_budget_can_never_exceed_what_the_driver_allows(self):
        _run_encoding(ENC_ROOM + r"""
assert.strictEqual(ENC.capacity(both, "astc-6x6-rgba", {astc: 192, rgba8: 48, maxLayers: 2048}), 128,
                   "the manifest's own budget");
assert.strictEqual(ENC.capacity(both, "webp-rgba", {astc: 192, rgba8: 48, maxLayers: 2048}), 48,
                   "the uncompressed cap: 128 RGBA8 layers would be 117 MB");
assert.strictEqual(ENC.capacity(both, "astc-6x6-rgba", {astc: 192, rgba8: 48, maxLayers: 32}), 32,
                   "MAX_ARRAY_TEXTURE_LAYERS is the last word");
assert.strictEqual(ENC.capacity({selection: {phone_budget: 400}}, "astc-6x6-rgba",
                                {astc: 192, rgba8: 48, maxLayers: 2048}), 192);
assert.strictEqual(ENC.capacity({}, "astc-6x6-rgba", {astc: 192, rgba8: 48, maxLayers: 2048}), 128,
                   "no budget in the manifest: the documented default");
assert.strictEqual(ENC.capacity(both, "astc-6x6-rgba", {maxLayers: 0}), 1, "never zero layers");
""")

    def test_the_page_uses_the_unit_and_says_which_side_lacks_astc(self):
        text = _template()
        assert "function chooseEncoding(man){ return ENC.choose(man, {astc: !!ext" in text
        assert "capacity = ENC.capacity(man, encoding, {" in text
        caption = _section(text, "function updateCaption(", "/* -------- verification hooks")
        # A Tower built without an encoder used to be reported as a phone
        # without one, so a Tower bug was going to be filed against a phone
        # (review 2, P-8).
        assert "this device has no compressed-texture support" in caption
        assert "the Tower built no compressed textures for this world" in caption
        assert "encoding_notes" in caption


class TestBootingWithNothingPlaced:
    """Review 2, M-0: a boot that placed no imagery used to be permanently dead
    on BOTH sides at once, and no control reached it."""

    def test_a_boot_with_no_layers_keeps_asking_instead_of_failing(self):
        text = _template()
        start = _section(text, "/* -------- start ---", "main().catch(")
        assert "if (!S.layers){" in start
        assert "dropAppearance(" in start, "the same not-terminal path a withdrawal takes"
        assert 'fail("No image' not in text, "`fail` stops the script above `follow()`"
        empty = start[start.index("if (!S.layers){"):start.index("await finishOpening();")]
        assert "follow();" in empty, "a page with nothing placed still asks again"

    def test_a_page_that_recovers_from_that_finishes_its_opening(self):
        """The recovery needs more than textures: a page that never chose an
        opening pose has `shownAt === null`, and `frame()` returns before it
        draws anything at all."""
        text = _template()
        poll = _section(text, "async function pollOnce(", "async function follow(")
        assert "if (shownAt === null) await finishOpening();" in poll
        opening = _section(text, "async function finishOpening(", "/* -------- start ---")
        for step in ("opening = await chooseOpening(", "setPose(restored || poseOf(ci));",
                     'S.phase = "ready";', "startNav();"):
            assert step in opening, step


def _code_only(section):
    """`section` with its comments removed.

    The page is commented far more heavily than it is coded, and the comments
    quote the very names the assertions below look for (`fail()`,
    `dropAppearance(...)`). Counting call sites has to count CODE.
    """
    out, in_block = [], False
    for line in section.splitlines():
        stripped = line.strip()
        if in_block:
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("/*"):
            in_block = "*/" not in stripped
            continue
        if stripped.startswith("//"):
            continue
        out.append(line)
    return "\n".join(out)


# A boot whose FIRST manifest fetch 404s, as the routes actually word it. The
# page has just called `dropAppearance`, so it holds no revision and nothing
# is being held on screen.
BOOT_GAP_SCRIPT = r"""
const page = {revision: null, holdingSince: null, now: 0};
// The gap, in the manifest route's own sentences (`AppearanceNotServed`).
// WORLD-BUILDER-WORLDS.md 4a rule 5: a 404 carrying one of 4's own sentences
// MAY BE TRANSIENT -- keep the picture and ask again at the next interval.
for (const detail of ["appearance is stale against the session's redaction record",
                      "no appearance for this session",
                      "session 's1' of world 'w1' has no geometry yet"]){
  const r = FOLLOW.decide({absent: detail}, page);
  assert.strictEqual(r.action, "none", "a transient 404 is not terminal: " + detail);
  const d = FOLLOW.nextDelay(r, FOLLOW.BASE_MS);
  assert.ok(d >= FOLLOW.BASE_MS && d <= FOLLOW.CEILING_MS, "slower, never silent: " + d);
}
// The revision route, meanwhile, says the final build is running.
const gap = {live: true, appearance: {revision: null, current: false, state: "rebuilding", epoch: null}};
assert.strictEqual(FOLLOW.decide({ok: gap}, page).action, "hold");
// Then the artifact appears, and the page that booted into nothing upgrades.
const served = {live: false, appearance: {revision: "s1/appearance:b2", current: true,
                                          state: "served", epoch: "b2"}};
const r = FOLLOW.decide({ok: served}, page);
assert.strictEqual(r.action, "load", "the boot-404 page loads the build when it is served");
assert.strictEqual(r.revision, "s1/appearance:b2");
assert.strictEqual(FOLLOW.nextDelay(r, FOLLOW.CEILING_MS), FOLLOW.BASE_MS, "back to the base rate");
assert.strictEqual(FOLLOW.mustReplace(null, {epoch: "b2"}), false, "nothing on screen to replace");
"""

# The other half of that rule, unchanged by the boot recovery.
GONE_SCRIPT = r"""
const page = {revision: null, holdingSince: null, now: 0};
for (const detail of ["no world 'w1'", "world 'w1' has no session 's1'"]){
  const r = FOLLOW.decide({absent: detail}, page);
  assert.strictEqual(r.action, "drop", "a world or session that is GONE still drops: " + detail);
  assert.ok(/no longer on the Tower/.test(r.reason), r.reason);
}
// and it is never confused with the sentences a boot 404 recovers from
for (const detail of ["appearance is stale against the session's redaction record",
                      "no appearance for this session",
                      "the render revision is not served"]){
  assert.strictEqual(FOLLOW.decide({absent: detail}, page).action, "none", detail);
}
// A page with imagery in hand is dropped on GONE whatever it was holding.
assert.strictEqual(FOLLOW.decide({absent: "no world 'w1'"},
                                 {revision: "s1/appearance:b1", holdingSince: 0, now: 0}).action,
                   "drop");
"""


class TestBootingIntoTheRebuildingGap:
    """Mac validation, T5: the page died permanently if its FIRST manifest fetch
    404ed. `fetchBytes` tags a 404 `e.absent`, `loadRevision` guards only
    `applyManifest`, so the boot `catch` called `fail()` -- which hides the
    page, sets `phase: failed`, and returns ABOVE both `follow()` call sites.
    `follow()` is the only revision poll there is, so the page could never
    discover the imagery when it appeared. A boot 404 is the ordinary case:
    Stop flips the session label and the manifest route raises until the
    rebuild lands (WORLD-BUILDER-APPEARANCE.md 9, "Nothing is SERVED during
    `rebuilding`"; WORLD-BUILDER-WORLDS.md 4a rule 5: a 404 with one of 4's
    own sentences may be transient -- keep asking). The `webglcontextrestored`
    handler had answered `e.absent` this way all along; the boot was the only
    path that treated it as terminal."""

    def test_a_boot_404_drops_and_keeps_asking_instead_of_failing(self):
        text = _template()
        start = _section(text, "/* -------- start ---", "main().catch(")
        code = _code_only(start)
        # The boot's first manifest fetch is guarded, and only an ABSENCE is
        # recovered -- a real fault still falls through to the boot catch.
        boot = code[code.index('status("Loading the images'):code.index("if (!S.layers){")]
        assert "await serial(() => loadRevision(CONFIG.appearance_revision || null));" in boot
        assert "} catch (e){" in boot and "if (!(e && e.absent)) throw e;" in boot
        assert "waitForTheTowerToServeTheImages();" in boot and "return;" in boot
        # The recovery itself: the withdrawal's mechanism, not its wording.
        wait = _section(text, "function waitForTheTowerToServeTheImages(){", "  try {")
        assert "dropAppearance(" in wait, "the same not-terminal path a withdrawal takes"
        assert "follow();" in wait, "and the page's existing poll, so it recovers on its own"
        assert "fail(" not in wait
        assert "no longer served" not in wait, "the common cause is 'not finished yet'"
        assert "it may still be being " in wait and '+ "finished.' in wait
        # NO `e.absent` PATH REACHES `fail(`: one call site is left in the boot,
        # the last resort in the catch, and the absence is routed out above it.
        assert code.count("fail(") == 1, "one last resort, for what cannot be followed"
        assert code.count("waitForTheTowerToServeTheImages();") == 2, "both absent paths"
        catch = code[code.rindex("} catch (e){"):]
        assert catch.index("if (e && e.absent){") < catch.index("fail("), (
            "an absence is routed to the recovery before `fail` is ever considered")
        assert "waitForTheTowerToServeTheImages(); return;" in catch
        # And no second timer: the recovery reuses `follow()`, nothing else.
        assert "setInterval(" not in code

    def test_the_recovered_boot_finishes_its_opening_when_a_build_arrives(self):
        """The boot 404 lands in exactly the state review 2's M-0 boot lands in
        -- `phase: withdrawn`, `shownAt === null` -- so `pollOnce`'s load branch
        is what finishes the boot for it too."""
        text = _template()
        poll = _section(text, "async function pollOnce(", "async function follow(")
        assert 'if (S.layers && (S.phase === "withdrawn" || bootFailed)){' in poll
        assert "if (shownAt === null) await finishOpening();" in poll
        drop = _section(text, "function dropAppearance(", "/* -------- following")
        assert 'S.phase = "withdrawn";' in drop and "S.revision = null;" in drop
        assert "holdingSince = null;" in drop
        follow = _section(text, "async function follow(", "S.refresh =")
        assert 'if (S.phase === "failed" && !bootFailed) return;' in follow, (
            "`failed` still ends the follower, except for the boot failure that "
            "is waiting for exactly the answer this loop asks for")

    def test_a_page_that_booted_into_the_gap_loads_the_build_when_it_is_served(self):
        """The upgrade itself, under node: the transient 404s change nothing,
        the gap holds, and the first served revision is a `load`."""
        _run_follower(BOOT_GAP_SCRIPT)

    def test_a_gone_shaped_404_is_still_terminal_for_the_imagery(self):
        """No regression on the other half of that rule. Note what "terminal"
        means here: `decide` returns `drop`, the textures go and the page says
        the world is no longer on the Tower. The poll loop itself does NOT
        stop -- every drop keeps asking, backing off -- and the boot recovery
        leaves that unchanged."""
        _run_follower(GONE_SCRIPT)


# A boot that failed for a fault that is NOT an absence: the page holds no
# revision (nothing was ever placed) and `fail()` has run.
BOOT_FAULT_SCRIPT = r"""
const page = {revision: null, holdingSince: null, now: 0};
// A transport error, a 500, three exhausted retries: `pollOnce` turns all of
// them into `{error: ...}`. WORLD-BUILDER-IOS.md's following table: "any other
// 404, another status, or a transport error: keeps the picture and asks again
// next interval."
let delay = FOLLOW.BASE_MS;
for (const message of ["Failed to fetch",
                       "the appearance manifest: HTTP 500",
                       "could not fetch the appearance manifest: no answer in 45 s"]){
  const r = FOLLOW.decide({error: message}, page);
  assert.strictEqual(r.action, "none", "a fault is not an answer: " + message);
  delay = FOLLOW.nextDelay(r, delay);
  assert.ok(delay >= FOLLOW.BASE_MS && delay <= FOLLOW.CEILING_MS, "backoff, never silent: " + delay);
}
assert.strictEqual(FOLLOW.nextDelay({action: "none", live: null}, FOLLOW.CEILING_MS),
                   FOLLOW.CEILING_MS, "the existing 120 s ceiling still applies");
// Then the Tower serves a build, and the page that failed to boot loads it.
const served = {live: false, appearance: {revision: "s1/appearance:b2", current: true,
                                          state: "served", epoch: "b2"}};
const r = FOLLOW.decide({ok: served}, page);
assert.strictEqual(r.action, "load", "a failed boot still upgrades when a build is served");
assert.strictEqual(r.revision, "s1/appearance:b2");
assert.strictEqual(FOLLOW.nextDelay(r, FOLLOW.CEILING_MS), FOLLOW.BASE_MS);
assert.strictEqual(FOLLOW.mustReplace(null, {epoch: "b2"}), false, "nothing on screen to replace");
"""


class TestABootFaultThatIsNotAnAbsence:
    """Mac validation, 2026-09-22 (follow-up). The 404 recovery above does not
    cover a transport error, a 500 or three exhausted retries: those still
    reach the boot `catch`, still call `fail()`, and `follow()` returns on
    `phase: failed`, so the page is again left with no poll of any kind. The
    app cannot rescue it -- a page that called `fail()` reports `didFinish`, so
    the screen is `.ready` and the "Try again" control is not shown -- and
    WORLD-BUILDER-IOS.md's own following table answers this class of fault the
    other way: "any other 404, another status, or a transport error: keeps the
    picture and asks again next interval". So `fail()` still runs, the message
    still stands, and the follower is started anyway."""

    def test_a_boot_fault_still_fails_loudly_but_leaves_a_live_follower(self):
        text = _template()
        start = _section(text, "/* -------- start ---", "main().catch(")
        code = _code_only(start)
        catch = code[code.rindex("} catch (e){"):]
        # the absence is still routed out first, and `fail` still runs after it
        assert catch.index("if (e && e.absent){") < catch.index("fail(")
        assert 'fail("The images could not be placed: "' in catch, (
            "this is about recovery, not about hiding the error")
        assert 'if (S.phase === "failed") return;' in catch, "never fail twice"
        # and the follower is started for it
        assert "if (glReady){ bootFailed = true; follow(); }" in catch
        assert catch.index("fail(") < catch.index("bootFailed = true"), (
            "the honest message is on screen before anything else happens")

    def test_only_a_page_that_could_draw_gets_a_follower(self):
        """Constraint: a genuinely un-followable fault must not spin a poller.
        `glReady` is the witness, and it is raised only once the GL programs,
        the basis and the recorded walk are all in hand."""
        text = _template()
        start = _section(text, "/* -------- start ---", "main().catch(")
        code = _code_only(start)
        assert "let glReady = false;" in code
        preamble = code[code.index("buildGL();"):code.index("glReady = true;")]
        for step in ("buildGL();", "basis();", "recordedPitch();"):
            assert step in preamble, step
        assert code.index("glReady = true;") < code.index('S.phase = "loading";')
        assert code.count("glReady = true;") == 1, "one witness, one place"
        assert code.count("bootFailed = true") == 1, "one flag, one place"
        # and the follower is started ONLY behind that witness
        catch = code[code.rindex("} catch (e){"):]
        assert "if (glReady){ bootFailed = true; follow(); }" in catch
        assert catch.count("follow();") == 1, "no ungated start"
        # no WebGL 2 at all never reaches this block: it fails and returns
        # above, where neither `glReady` nor `follow` exists yet
        gl = _section(text, 'let gl = canvas.getContext("webgl2", attrs);', "const cams =")
        assert "needs WebGL 2" in gl and "return;" in gl
        assert "follow" not in gl and "bootFailed" not in gl
        # and a fault outside `main`'s own try is still terminal
        tail = text[text.index("main().catch("):]
        assert "fail(" in tail and "follow(" not in tail

    def test_the_flag_cannot_outlive_the_boot_it_was_set_for(self):
        """The one way a flag like this goes wrong is by keeping a page alive
        that was later failed for a reason nothing can follow. It is cleared by
        the first build that opens, so from then on the ordinary rules hold --
        a restore that cannot replace the imagery still ends the page."""
        text = _template()
        reopen = _section(text, "function reopenAfterBootFailure(){", "async function pollOnce(")
        assert "bootFailed = false;" in reopen
        # it undoes exactly what `fail()` hid, and nothing else
        hid = _section(text, "function fail(text){", "/* THE STATUS LINE SITS ABOVE")
        for el in ("wrap", "bar", "caption"):
            assert f'$("{el}").style.display = "none";' in hid or f'$("{el}").style.display = "none"' in hid, el
            assert f'$("{el}").style.display = "";' in reopen, el
        poll = _section(text, "async function pollOnce(", "async function follow(")
        assert 'if (S.layers && (S.phase === "withdrawn" || bootFailed)){' in poll, (
            "only with layers in hand: it can never un-hide an empty page")
        assert "if (bootFailed) reopenAfterBootFailure();" in poll
        assert poll.index("reopenAfterBootFailure();") < poll.index("await finishOpening();")
        # A context-restore failure is followable ONLY after the GL side came
        # back (review, 2026-09-23): a `buildGL()` that throws stays terminal,
        # and a data-path fault after it keeps the poller and forgets the
        # revision so the next answer is a `load` rather than "unchanged".
        restored = _section(text, 'canvas.addEventListener("webglcontextrestored"', "/* -------- input")
        assert "fail(" in restored
        assert restored.index("buildGL();") < restored.index("rebuilt = true;") < (
            restored.index("await loadRevision(S.revision);"))
        assert "if (rebuilt){ bootFailed = true; S.revision = null; }" in restored
        assert restored.index("fail(") < restored.index("if (rebuilt){ bootFailed = true;")

    def test_a_failed_boot_keeps_asking_and_loads_the_build_when_it_comes(self):
        """Under node: the faults change nothing and back off to the existing
        ceiling, and the first served revision is a `load`."""
        _run_follower(BOOT_FAULT_SCRIPT)


class TestNothingBlocksTheMainThreadUnbounded:
    """Review 2, P-3 and P-4. Measured before and after in headless Chrome
    (`Glasses-scratch/wb-final-recon/fixit/fix-ios2/stats/boot_*.json`): the
    longest main-thread block during boot fell from 4.6 s to 0.6 s. These are
    the structural facts that keep it that way."""

    def test_the_per_keyframe_depth_pass_yields_on_a_time_budget(self):
        text = _template()
        src = _section(text, "async function renderSourceDepth(", "function renderPendingSourceDepth(")
        assert "if (performance.now() - sliceStart > SOURCE_DEPTH_SLICE_MS){" in src
        assert "await new Promise(res => setTimeout(res, 0));" in src
        # what a frame drawn in the gap would have changed, put back
        after_yield = src.split("sliceStart = performance.now();")[2][:500]
        assert "gl.bindFramebuffer(gl.FRAMEBUFFER, fbSrc);" in after_yield
        assert "gl.depthMask(true); gl.colorMask(true, true, true, true);" in src
        # and a lost context or a cleared texture set stops it
        assert "if (!G || !G.dep || !G.col) break;" in src
        # one pass at a time, because three upload workers ask for it
        chain = _section(text, "function renderPendingSourceDepth(", "/* -------- loading and live append")
        assert "depthChain.then(" in chain
        assert text.count("await renderPendingSourceDepth();") == 2

    def test_the_opening_scan_yields_on_a_time_budget(self):
        text = _template()
        opening = _section(text, "async function chooseOpening(", "/* -------- context loss")
        assert "if (performance.now() - sliceStart > OPENING_SLICE_MS){" in opening
        assert "const s = await score(i);" in opening
        assert "opening = await chooseOpening(" in text

    def test_the_navigation_input_is_a_stepper_driven_by_the_same_budget(self):
        text = _template()
        assert "function* navInputSteps(" in text and "function* voidEdgesSteps(" in text
        inp = _section(text, "function* navInputSteps(", "/* Large holes in the proxy")
        assert inp.count("yield") >= 5, "every heavy loop in it yields"
        assert "yield* voidEdgesSteps();" in inp, "the hole scan is sliced with it"
        build = _section(text, "function buildNav(){", "/* Overview: the best-supported")
        assert "do { r = steps.next(); } while (!r.done && performance.now() - t < 14);" in build, (
            "the input gets the same 14 ms budget the field work already had")
        assert "navInput()" not in text, "nothing calls the old unsliced form"


# ---------------------------------------------------------------------------
# fix-it ux lane: the cold open, the finger that arrives first, the dark that
# is a state rather than a toast, what a body can reach, and the haze the
# caption has always promised.
# Measured in `Glasses-scratch/wb-final-recon/fixit/ux/` (UX.md), against
# VISUAL-REVIEW-3 at 947525f.
# ---------------------------------------------------------------------------


class TestTheColdOpenIsLegibleFromTheFirstPaint:
    """VISUAL-REVIEW-3 §1, BLOCKING. Every cold open showed the finished
    caption and the whole button bar over a pure black canvas, with the status
    line cleared, for 1.45-1.71 s on the reviewer's machine and 2.8-3.1 s on
    this one. Three things had to be true and now are: the chrome does not
    arrive before the picture, the page never falls silent while it is still
    working, and the picture itself arrives far sooner."""

    def test_the_chrome_is_held_until_there_is_something_to_see(self):
        text = _template()
        assert '<body class="booting">' in text
        assert "body.booting #caption,body.booting #bar{opacity:0;pointer-events:none}" in text
        first = _section(text, "  function showFirst(){", "  let navStarted = false;")
        assert 'document.body.classList.remove("booting");' in first
        assert first.index("shownAt =") < first.index('classList.remove("booting")')
        # and it is the FIRST DRAWN FRAME that takes it off, not a timer
        assert "frame(true);" in first

    def test_the_page_never_falls_silent_while_it_is_still_working(self):
        text = _template()
        assert 'const CHOOSING = "Choosing where to open…";' in text
        apply_ = _section(text, "async function applyManifest(", "let loadGeneration = 0;")
        assert 'status(shownAt === null ? CHOOSING : "");' in apply_, \
            "the manifest landing does not clear the status while nothing is drawn"
        assert 'status("");' not in apply_.split("manifest = man;")[1], "and nothing else clears it"
        first = _section(text, "  function showFirst(){", "  let navStarted = false;")
        assert "status(PREPARING);" in first

    def test_the_opening_is_placed_and_drawn_long_before_the_scan_finishes(self):
        text = _template()
        op = _section(text, "async function chooseOpening(", "/* -------- context loss")
        assert "const PROVISIONAL_AFTER = 6;" in text
        assert "if (!placed && onProvisional && seen.size >= PROVISIONAL_AFTER" in op
        # the first few candidates are spread over the WHOLE walk, so what the
        # page opens on is already close to what it settles on
        assert "const order = [];" in op and "bit-reversed" in op.lower().replace("-", "-")
        # and the scoring camera is restored with NO yield in between, because
        # the page is drawing now and an animation frame must never catch it
        sc = op[op.index("const score = async i =>"):op.index("let best = -1")]
        assert sc.index("await new Promise") < sc.index("const keepCam = cam")
        assert "cam = keepCam; override = keepOverride;" in sc
        assert "await" not in sc[sc.index("const keepCam = cam"):]

    def test_the_rescue_is_ready_when_the_camera_is(self):
        """*Best view* needs the fine field and the review timed it at 7.8-8.0 s
        from the tap. *Reset* needs only an opening pose, so it is enabled at
        the first drawn frame -- and honestly disabled before it."""
        text = _template()
        assert '$("bReset").disabled = true;' in text
        first = _section(text, "  function showFirst(){", "  let navStarted = false;")
        assert '$("bReset").disabled = false;' in first
        # the field starts beside the rest of the scan rather than after it
        assert "function startNav(){ if (navStarted) return; navStarted = true; buildNav(); }" in text
        assert "startNav();" in first


class TestAFingerThatArrivesBeforeThePicture:
    """VISUAL-REVIEW-3 §2, BLOCKING. Input during `loading` went into `ctl`,
    where nothing consumed it because `frame()` draws nothing before the
    opening is placed -- and then ALL of it ran in the first frame after the
    handover. A thumb resting on the glass through the load left the camera at
    yaw 2.90 rad in the empty half of the room, and a look the person made is
    deliberately never taken back, so it stayed there."""

    def test_input_before_the_first_frame_is_deferred_and_never_banked(self):
        text = _template()
        assert 'const TOUCHED_EARLY = "One moment — you can look around as soon as it draws";' in text
        d = _section(text, "  function deferInput(){", "  function feel(")
        assert "if (shownAt !== null) return false;" in d
        assert "hint(0.6, TOUCHED_EARLY);" in d, "it is said, not silently dropped"
        assert "ctl.look = [0, 0]; ctl.move = [0, 0, 0];" in d and "vel.look = [0, 0];" in d
        # every way in goes through it
        for fn, start, end in (
            ("feel", "  function feel(look, move, now){", "  function interrupt("),
            ("wheel", 'canvas.addEventListener("wheel"', "addEventListener(\"keydown\""),
            ("keys", 'addEventListener("keydown"', 'addEventListener("keyup"'),
            ("S.input", "  S.input = (c) => {", "  S.step = (d) =>"),
        ):
            assert "deferInput()" in _section(text, start, end), fn
        down = _section(text, 'canvas.addEventListener("pointerdown"',
                        'canvas.addEventListener("pointermove"')
        assert "if (shownAt === null) deferInput();" in down

    def test_a_view_the_reader_aimed_survives_the_handover(self):
        text = _template()
        fin = _section(text, "async function finishOpening(", "/* -------- start ---")
        assert "&& !touched){" in fin, "the correction only happens if nobody has aimed it"
        assert 'S.openingKept = touched ? "the reader\'s own view" : "the provisional";' in fin
        assert "let touched = false;" in text
        assert "touched = true;" in _section(text, "  function feel(look, move, now){",
                                             "  function interrupt(")


class TestTheDarkIsSaidForAsLongAsItIsTrue:
    """VISUAL-REVIEW-3 §3, BLOCKING. `hint()` showed the dark sentence for
    1,100 ms and `hintSaid` throttled it to once per 4,000 ms, so over a
    24-step turn from the opening it was visible at three steps and at NONE of
    the six that render a 100% black frame. Measured again here after the
    change: visible at 10 of 24 and at 6 of 6 (UX.md §3)."""

    def test_the_dark_state_is_permanent_while_true_and_silent_when_not(self):
        text = _template()
        assert '<div id="dark" role="status" aria-live="polite"></div>' in text
        assert "#dark.on{opacity:.88;pointer-events:auto}" in text
        d = _section(text, "  const DARK_SAY_ON = 0.90", "  const KEY_SPEED")
        assert "const DARK_SAY_ON = 0.90, DARK_SAY_OFF = 0.45, DARK_SAY_MS = 320;" in d
        # hysteresis, so it cannot blink on the boundary
        assert "const on = darkShown ? dark > DARK_SAY_OFF : dark > DARK_SAY_ON;" in d
        # and a delay, so sweeping through a dark patch does not flash it
        assert "now - darkSince >= DARK_SAY_MS" in d
        # it is up to date on every drawn frame
        frame = _section(text, "function frame(sync){", "function drawnFraction(")
        assert "updateDark(dk);" in frame
        # the throttle is gone entirely
        assert "hintSaid" not in text

    def test_the_dark_line_is_also_a_way_out(self):
        text = _template()
        assert '$("dark").onclick = () => faceTheRoom();' in text


class TestWhatABodyCanReach:
    """VISUAL-REVIEW-3 §6, MAJOR. A finger got forward 100% and right 100% of
    what it asked -- those run ALONG the walk, where the tube does not bind --
    against backward 31% and up and down 16%. The tube was never measured
    against the imagery; it is now (UX.md §4), and the imagery holds well past
    where it stopped you."""

    def test_the_tube_is_wide_enough_that_every_direction_answers(self):
        _run_nav(r"""
const push = (d, n) => {
  let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1}, worst = 0;
  for (let i = 0; i < n; i++){
    cam = NAV.step(F, path, cam, {look: [0, 0], move: d, held: true}, 16, V);
    worst = Math.max(worst, cam.resisted);
  }
  return {gone: dist(cam.p, at), resisted: worst, p: cam.p};
};
const up_ = push([0, 0.1, 0], 40), dn = push([0, -0.1, 0], 40);
const bk = push([0, 0, -0.1], 40), lf = push([-0.1, 0, 0], 40), rt = push([0.1, 0, 0], 40);
// every direction MOVES, and by something a body would notice
for (const [n, r] of [["up", up_], ["down", dn], ["back", bk]])
  assert.ok(r.gone > 1.0, n + " went nowhere: " + r.gone);
// up and down are the same limit, not two different ones
assert.ok(Math.abs(up_.gone - dn.gone) < 0.05, "up and down are symmetric: " + up_.gone + " " + dn.gone);
// and each of them STOPS, visibly: the resistance the edge hint is raised on
for (const [n, r] of [["up", up_], ["down", dn], ["back", bk]])
  assert.ok(r.resisted > 0.8, n + " stopped without saying so: " + r.resisted);
// along the walk nothing binds at all, which is why forward and right were
// already 100% and the complaint was about the other four
assert.ok(rt.gone > 3.5 && rt.gone > 1.5 * bk.gone,
          "along the walk is far freer than across it: " + rt.gone + " vs " + bk.gone);
// the vertical limit is the one that moved, and it is a real fraction of the
// scene rather than a tenth of it
assert.ok(NAV.V_MAX >= 1.4 && NAV.R_MAX >= 2.2, "the tube: " + NAV.R_MAX + " x " + NAV.V_MAX);
assert.ok(NAV.R_SOFT / NAV.R_MAX <= 0.51, "half the tube is still free of resistance");
""")

    def test_the_lattice_pays_for_the_wider_tube(self):
        """The field's volume grew with the tube; the fine pass is what *Best
        view* waits for, so the spacing is what pays rather than the wait.
        Measured on the canonical world: 3,851 voxels at 0.5 over the old tube,
        5,914 at 0.6 and 4,259 at 0.7 over the new one."""
        nav = _nav_source()
        assert "const SPACING = 0.7 * UNIT," in nav


class TestTheBestViewStaysWhereItPutsYou:
    """VISUAL-REVIEW-3 §6, MAJOR ("the page leaves its own tube"). It never
    did leave the tube -- the destination measured at envelope 0.52-0.78 of 1
    -- but 0.5 is where a RELEASED camera starts drifting back, so the page
    could fly you somewhere it would then pull you out of."""

    def test_the_destination_is_inside_the_part_of_the_tube_that_does_not_drift(self):
        text = _template()
        over = _section(text, "const OVERVIEW_AWAY", "function overview(){")
        assert "let maxE = NAV.R_SOFT / NAV.R_MAX;" in over
        assert "if (pd.e > maxE || !pd.closest) continue;" in over
        # and it is a preference, never a dead button
        assert "while (!found.length && maxE < 1){" in over
        _run_nav(r"""
// at the free edge the drift is exactly nothing, which is the property the
// filter is buying
assert.strictEqual(NAV.positionBound(NAV.R_SOFT / NAV.R_MAX), 0);
assert.ok(NAV.positionBound(0.8) > 0.3, "and past it there really is a drift");
""")


class TestTheCompassIsLegibleAtRest:
    """VISUAL-REVIEW-3 §12, MINOR. It had no label, no legend, and its opacity
    ramped in only once the support field existed -- a second or more after
    the first picture, which is exactly when a first-time viewer looks at it."""

    def test_it_is_drawn_from_the_first_frame_and_it_says_what_it_is(self):
        text = _template()
        assert '<div id="clabel" aria-hidden="true">photographed<br>from here</div>' in text
        cue = _section(text, "  const RING_R = 21, RING_SIZE = 58;", "  /* The way back.")
        assert "if (shownAt === null){ el.classList.remove(\"on\"); lab.classList.remove(\"on\"); return; }" in cue
        # without a field the arcs are blank, which is the truth, not hidden
        assert "const k = ringProf ? NAV.smooth(0.08, NAV.T_HI, ringProf[i]) : 0;" in cue
        assert 'g.fillText("YOU", c, c);' in cue

    def test_the_way_home_does_not_change_its_mind_at_the_antipode(self):
        text = _template()
        cue = _section(text, "  const BACK_ON = 0.60", "  /* THE DARK STATE")
        assert "const BACK_FLIP = 0.35;" in cue
        assert "const ambiguous = Math.PI - Math.abs(b.delta) < BACK_FLIP;" in cue
        assert "if (!backSide || !ambiguous) backSide = side;" in cue


class TestTheSmallerComplaints:
    """VISUAL-REVIEW-3 §7, §12, §13."""

    def test_placing_the_camera_clears_the_input_as_well_as_the_inertia(self):
        text = _template()
        for fn, start, end in (("setPose", "  function setPose(pose){", "  function glideTo("),
                               ("glideTo", "  function glideTo(pose, index){", "  /* The hint:")):
            s = _section(text, start, end)
            assert "ctl.look = [0, 0]; ctl.move = [0, 0, 0];" in s, fn
            assert "vel.look = [0, 0]; vel.move = [0, 0, 0];" in s, fn

    def test_a_placed_camera_reports_a_yaw_a_person_could_read(self):
        text = _template()
        setp = _section(text, "  function setPose(pose){", "  function glideTo(")
        assert "Math.atan2(Math.sin(pose.yaw), Math.cos(pose.yaw))" in setp
        nav = _section(text, "  function navUpdate(dt){", "  /* The envelope's support field")
        assert "cam.yaw = Math.atan2(Math.sin(cam.yaw), Math.cos(cam.yaw));" in nav

    def test_a_press_of_the_arrow_covers_ground(self):
        """0.049 scene units a press is not stepping through a walk. A press
        now advances until it has gone STEP_UNITS of path or passed
        STEP_MAX_POSES, and the quality rule decides where in that run it
        lands. Measured on the canonical world from the review's own starting
        index: 0.165 units a press before, 0.308 after (UX.md §7)."""
        text = _template()
        assert "const STEP_MAX_POSES = 12;" in text
        assert "function stepUnits(){ return 0.32 * ((CONFIG.median_scene_depth || 4.7) / 4.7); }" in text
        wt = _section(text, "  function walkTarget(from, d){", "  function step(d){")
        assert "if (j < 0 || j >= cams.length) break;" in wt, "both ends of the walk stay inert"
        assert "if (gone >= want) break;" in wt
        st = _section(text, "  function step(d){", "  let poseQ")
        assert "if (to === ci){ hint(0.6); return; }" in st

    def test_the_about_panel_is_sections_not_a_wall(self):
        """335 words in one block, half the height of a phone screen. The words
        are nearly the same; the shape is not."""
        text = _template()
        cap = _section(text, "function updateCaption(", "/* -------- verification hooks")
        assert "const sections = [" in cap
        assert cap.count('document.createElement("em")') == 1 and "for (const [title, body] of sections)" in cap
        titles = ["What you are looking at", "The flat grey patches", "Looking and moving",
                  "Finding your way", "The walk"]
        for title in titles:
            assert '["' + title + '"' in cap, title
        assert "#caption .more em{display:block" in text
        assert "max-height:44vh;overflow-y:auto" in text


class TestTheHazeAndWhatTheFadeIsFor:
    """VISUAL-REVIEW-3 §9 and §10. The caption promised a grey haze; the page
    drew the background, which the reviewer measured at luminance 10-18
    against a background of 11-28 -- darker than the emptiness it was supposed
    to be distinguished from. And the confidence fade, A/B'd at `clo=0&chi=0`,
    changed a median 0.22 of an 8-bit level over the whole frame: invisible,
    because it faded toward the same near-black. One change answers both."""

    def test_the_haze_is_flat_and_plainly_between_the_room_and_the_void(self):
        text = _template()
        shade = _section(text, "const GLSL_SHADE", "const FS_BLEND")
        assert "const vec3 HAZE_RGB = vec3(0.160, 0.168, 0.190);" in shade
        # brighter than the brightest point of the background it sits on
        bg = _section(text, "const GLSL_BACKGROUND", "`;")
        assert "vec3(0.078, 0.084, 0.094)" in bg
        assert min(0.160, 0.168, 0.190) > max(0.078, 0.084, 0.094)
        # NOT flat any more, and this is the whole of what changed: one
        # constant plus a deterministic screen-space hash of +/- 2 of 255,
        # because flat read as "a UI panel that failed to paint"
        # (VISUAL-REVIEW-4 #5). No texture fetch, no hue, zero mean, and
        # bounded: the amplitude is written down here as well as there.
        h = _section(shade, "float hazeHash(", "/* THE CONFIDENCE FADE")
        assert "fract(sin(dot(v" in h and "texture" not in h
        assert "return HAZE_RGB + vec3(n * (2.0 / 255.0));" in h
        assert "- 0.5" in h, "zero mean"

    def test_no_measurement_and_no_score_can_see_the_haze(self):
        """The opening and the Best view are scored on the drawn fraction, the
        honesty check reads the observed/unobserved mask, and the fade's cost
        is read through mode 11. All three return before the haze is drawn,
        and the uniform is zero for every mode but the display one."""
        text = _template()
        shade = _section(text, "const GLSL_SHADE", "const FS_BLEND")
        haze_at = shade.index("if (uHaze <= 0.0)")
        for mode in ("uMode == 2", "uMode == 3", "uMode == 11"):
            assert shade.index(mode) < haze_at, mode
        uni = _section(text, "function shadeUniforms(", "function blendPass(")
        assert "gl.uniform1f(P.u.uHaze, mode ? 0 : OPT.haze);" in uni

    def test_the_fade_gives_way_to_the_haze_so_that_it_can_be_seen(self):
        text = _template()
        shade = _section(text, "const GLSL_SHADE", "const FS_BLEND")
        assert "c = mix(uHaze > 0.0 ? haze() : background(), c, confKeep());" in shade
        # the band itself is UNCHANGED: raising it was measured in this lane
        # and rejected (2.53% of drawn at 110/255 for no visible gain, 6.85% at
        # 140/255, where it starts fogging real photographs)
        from tower.world_builder.appearance_render import CONFIDENCE_HI, CONFIDENCE_LO
        assert (CONFIDENCE_LO, CONFIDENCE_HI) == (24 / 255, 78 / 255)
        # and it is still evidence-preserving
        assert "o = vec4(c * alpha, alpha);" in shade

    def test_the_haze_can_be_turned_off_and_is_configured_from_one_place(self):
        from tower.world_builder.appearance_render import UNSEEN_HAZE

        text = _template()
        assert "haze: Math.max(0, Math.min(1, +(Q.get(\"haze\") ?? CONFIG.unseen_haze ?? 0.9)))" in text
        assert 0 < UNSEEN_HAZE <= 1


# ---------------------------------------------------------------------------
# fix-it bestview lane: the button lands in the same place twice, and the
# place it lands is chosen against the campaign's own measures
# ---------------------------------------------------------------------------


class TestTheBestViewLandsInTheSamePlaceEveryTime:
    """Twenty cold boots of the candidate build and twenty of the baseline
    each chose ONE destination, and the whole twelve-row candidate table came
    back identical to the last decimal in all forty (BESTVIEW.md §3): the
    chooser is not random. What it was sensitive to is the camera the reader
    left behind. `navView()` reports the VERIFICATION OVERRIDE's field of
    view whenever one is set, so a press after a `setView` at 70 degrees
    filtered the field through a frame the page never draws -- 97 candidates
    against a fresh page's 126 -- and landed somewhere else. That is why two
    lanes measured the same two builds and disagreed about where Best view
    goes (RIMS.md's table against CANDIDATE.md §8 #1)."""

    def _over(self):
        return _section(_template(), "const OVERVIEW_AWAY", "function overview(){")

    @staticmethod
    def _code_only(text):
        """The section with its comments taken out: this lane's comments name
        `navView()` several times and the order that matters is the code's."""
        import re as _re
        text = _re.sub(r"/\*.*?\*/", "", text, flags=_re.S)
        return _re.sub(r"(?m)^\s*//.*$", "", text)

    def test_the_chooser_clears_the_override_before_it_takes_its_view(self):
        body = self._code_only(self._over())
        body = body[body.index("function findOverview(){"):]
        assert body.index("override = null;") < body.index("navView()"), \
            "the candidate filter's field of view must be the page's own, not a setView's"
        assert body.count("navView()") == 1, "one view, taken once, with the override gone"
        # and it is put back on every way out, including the empty shortlist
        assert "const restore = () => { cam = save; override = saveOverride; };" in body
        assert "if (!top.length){ restore(); return null; }" in body
        assert body.count("restore();") >= 2

    def test_the_ranking_is_a_total_order_that_does_not_depend_on_input_order(self):
        over = self._over()
        src = over[over.index("function betterView"):over.index("function findOverview")]
        _run_plain(src + r"""
const A = {rendered: 0.8, yaw: 0.5, p: [0, 0, 0]};
const B = {rendered: 0.7, yaw: 0.1, p: [1, 1, 1]};
assert.ok(betterView(A, B) && !betterView(B, A), "a higher score wins, and only one way round");
assert.ok(betterView(A, null), "anything beats nothing");
assert.ok(!betterView(null, A) && !betterView(null, null));
assert.ok(!betterView(A, A), "and nothing beats itself");
// an exact tie is broken by the candidate's OWN numbers, not by where it sat
const T1 = {rendered: 0.5, yaw: 0.1, p: [0, 0, 0]};
const T2 = {rendered: 0.5, yaw: 0.2, p: [0, 0, 0]};
const T3 = {rendered: 0.5, yaw: 0.2, p: [0, 0, 1]};
assert.ok(betterView(T1, T2) && !betterView(T2, T1), "ties break on yaw");
assert.ok(betterView(T2, T3) && !betterView(T3, T2), "then on position");
// the same set in any order picks the same winner
const set = [A, B, T1, T2, T3, {rendered: 0.9, yaw: 0.3, p: [2, 0, 0]}];
const winnerOf = list => { let best = null; for (const f of list) if (betterView(f, best)) best = f; return best; };
const first = winnerOf(set);
for (let s = 1; s <= 200; s++){
  const shuffled = set.slice();
  for (let i = shuffled.length - 1; i > 0; i--){
    const j = (s * 7919 + i * 104729) % (i + 1);
    const t = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = t;
  }
  assert.strictEqual(winnerOf(shuffled), first, "shuffle " + s);
}
""")

    def test_the_press_reports_what_it_chose_and_what_it_chose_it_over(self):
        over = self._over()
        for key in ("chosenIndex:", "runnerUpIndex:", "runnerUpPose:", "margin:", "probe:"):
            assert key in over, key


class TestTheBestViewAvoidsAFrameThatIsTornToPieces:
    """The confidence term catches a candidate whose geometry the fade eats.
    It did not catch the frame the owner would have seen: `faded` 0.40%,
    98.8% drawn, detail above the reference, the widest depth range in the
    set -- and a stair-stepped black shape with a column of bright cream
    fragments down its right quarter. What separates it is how BROKEN UP the
    missing part is, which is what `torn` measures (BESTVIEW.md §4)."""

    def _over(self):
        return _section(_template(), "const OVERVIEW_AWAY", "function overview(){")

    def test_the_term_is_bounded_monotone_and_saturates(self):
        over = self._over()
        src = over[over.index("const OVERVIEW_TORN_REF"):over.index("function betterView")]
        _run_plain(src + r"""
assert.strictEqual(cleanTerm(0), 1, "a frame with no torn edge at all pays nothing");
assert.strictEqual(cleanTerm(undefined), 1, "and neither does one that was not measured");
assert.ok(Math.abs(cleanTerm(OVERVIEW_TORN_REF) - OVERVIEW_CLEAN_FLOOR) < 1e-12);
assert.ok(Math.abs(cleanTerm(1) - OVERVIEW_CLEAN_FLOOR) < 1e-12, "and it saturates, never below");
let prev = 2;
for (let t = 0; t <= 0.1; t += 0.0005){
  const c = cleanTerm(t);
  assert.ok(c <= prev + 1e-12 && c >= OVERVIEW_CLEAN_FLOOR - 1e-12, "t " + t);
  prev = c;
}
""")

    def test_the_term_is_in_the_score_and_the_speckle_proxy_is_not(self):
        over = self._over()
        assert "* cleanTerm(r.torn);" in over, "the clean term multiplies the rendered score"
        assert "cleanTerm(f.torn)" in over, "and the same number is reported"
        # `speck` is measured and reported beside it, and deliberately unscored:
        # at the probe size it does not reproduce the campaign's speckle, and
        # penalising it picks a frame that measures cleaner and looks worse.
        assert "speck: +(f.speck || 0).toFixed(5)" in over
        assert "cleanTerm(r.torn," not in over and "cleanTerm(f.speck" not in over
        # the statistics come out of the render the score already pays for
        clean = _section(_template(), "HOW UGLY the frame is",
                         "    // How much of what this view draws")
        assert "if (wantClean){" in clean and "drawBlend(" not in clean

    def test_on_the_measured_candidates_it_moves_the_destination_off_the_speckle(self):
        """The decisive rows of the twelve-candidate tables this lane measured
        on both builds at the shipping probe (BESTVIEW.md §4.4). On the
        candidate build c11 is the frame with the cream-speckle column --
        0.70% black and 0.330% speckle at 900x700, the worst speckle in the
        set -- and c05 is the frame that measures best on black, seam and
        detail. The old score prefers c11; with the clean term c05 wins. On
        the baseline the same rule leaves the destination exactly where it
        already was, which is the point: the term is aimed at the regression
        and at nothing else."""
        over = self._over()
        src = over[over.index("const OVERVIEW_TORN_REF"):over.index("function betterView")]
        _run_plain(src + r"""
const ref = 4.121, DETAIL_REF = 0.12, FADED_REF = 0.05, SOUND_FLOOR = 0.45;
const dterm = d => Math.max(0, Math.min(1, d / DETAIL_REF));
const sound = f => SOUND_FLOOR + (1 - SOUND_FLOOR) * (1 - Math.min(1, f / FADED_REF));
const base = t => t.drawn * (0.4 + 0.6 * Math.min(1, t.distance / (1.5 * ref)))
  * (0.5 + 0.5 * Math.max(0, t.toward)) * (0.15 + 0.85 * dterm(t.detail))
  * (0.5 + 0.5 * Math.min(1, t.range / ref)) * sound(t.faded);
// candidate build, at the shipping probe (`stats\shots_st_cand_128.json`)
const c11 = {drawn: 0.989, detail: 0.1059, range: 8.24, toward: 0.858, distance: 10.858,
             faded: 0.0040, torn: 0.02047};   // the cream-speckle column
const c05 = {drawn: 0.997, detail: 0.1136, range: 3.14, toward: 0.793, distance: 5.161,
             faded: 0.0039, torn: 0.00344};   // 0.01% black, seam 3.82, the most detail
const c00 = {drawn: 0.997, detail: 0.0975, range: 3.18, toward: 0.775, distance: 7.974,
             faded: 0.0012, torn: 0.00766};   // the runner-up once the term is in
assert.ok(base(c11) > base(c05), "the old score prefers the frame with the speckle column");
assert.ok(base(c05) * cleanTerm(c05.torn) > base(c11) * cleanTerm(c11.torn),
          "and the clean term puts it back");
assert.ok(base(c05) * cleanTerm(c05.torn) > base(c00) * cleanTerm(c00.torn),
          "over the runner-up too, and by a margin worth stating");
assert.ok(base(c05) * cleanTerm(c05.torn) / (base(c00) * cleanTerm(c00.torn)) > 1.10,
          "at least ten per cent, not a coin flip");
// baseline build, same probe: the destination does not move
const b05 = {drawn: 0.997, detail: 0.1126, range: 3.90, toward: 0.810, distance: 5.875,
             faded: 0.0047, torn: 0.00578};
const b01 = {drawn: 0.997, detail: 0.1135, range: 2.86, toward: 0.861, distance: 8.013,
             faded: 0.0027, torn: 0.00867};
const b02 = {drawn: 0.985, detail: 0.1030, range: 4.88, toward: 0.887, distance: 10.875,
             faded: 0.0122, torn: 0.01961};
assert.ok(base(b05) > base(b01) && base(b05) > base(b02), "it already won on the baseline");
for (const other of [b01, b02])
  assert.ok(base(b05) * cleanTerm(b05.torn) > base(other) * cleanTerm(other.torn),
            "and it still does");
""")


class TestThePageOpensSomewhereWorthLookingRoundFrom:
    """The opening scan scored the frame in FRONT of the camera and had no
    opinion about what happens when the reader looks round, which is their
    first move. At the pose it chose there was 150 degrees of horizon with
    nothing in it, where two other poses on the same walk have none; over all
    198 recorded poses it ranked 160th by mean horizon support
    (VISUAL-REVIEW-4 #1, BESTVIEW.md 6.1).

    The 36-bin ring would answer this exactly and cannot be used here: the
    ring needs the support field, and the field is started BY the opening
    scan. What the page has from its first frame is the recorded cameras, so
    SURROUND is the share of the horizon that any recorded camera within the
    tube of a pose pointed at."""

    SRC = ("const EMPTY_DETAIL", "/* Sliced for the same reason")

    # A synthetic walk in two clusters further apart than the tube. Every
    # camera in the first looks one way; the second is the same spot looked at
    # from eight evenly spaced directions.
    ROOM = r"""
const Q = new Map();                       // no ?osur=, so the shipping floor
const RING_BINS = 36, NAV = {R_MAX: 2.2};
const canvas = {width: 900, height: 700};
const CONFIG = {}, diag = 10;
const viewFovY = () => 1.12;
const yawPitchOf = d => ({yaw: Math.atan2(d[0], d[2]),
                          pitch: Math.asin(Math.max(-1, Math.min(1, d[1])))});
const cams = [];
const BLINKERED = [], ALLROUND = [];
for (let k = 0; k < 8; k++){                      // all looking +z
  BLINKERED.push(cams.length); cams.push([k * 0.1, 0, 0, 0, 0, 1]);
}
for (let k = 0; k < 8; k++){                      // looking eight ways
  const a = k * Math.PI / 4;
  ALLROUND.push(cams.length); cams.push([10 + k * 0.1, 0, 0, Math.sin(a), 0, Math.cos(a)]);
}
"""

    def test_the_surround_is_the_share_of_the_horizon_the_walk_pointed_at(self):
        src = _section(_template(), *self.SRC)
        _run_plain(self.ROOM + src + r"""
const blink = surroundOf(BLINKERED[3]), round = surroundOf(ALLROUND[3]);
// the two clusters are 10 units apart and the tube is 2.2, so neither sees
// the other: each pose is judged on the cameras a reader could walk to
assert.ok(round > 0.98, "eight evenly spaced looks cover the horizon: " + round);
assert.ok(blink < 0.4, "eight looks the same way cover one frame's worth: " + blink);
assert.ok(round > blink * 2, "and the ordering is the complaint's: " + blink + " " + round);
for (let i = 0; i < cams.length; i++){
  const s = surroundOf(i);
  assert.ok(s >= 0 && s <= 1, "a share of the horizon: " + i + " " + s);
}
""")

    def test_it_is_a_factor_with_a_high_floor_and_not_a_filter(self):
        """A good frame with a poor surround still beats a poor frame with a
        good one. The button's promise is still the best VIEW; the surround
        decides between views that are otherwise close."""
        src = _section(_template(), *self.SRC)
        _run_plain(self.ROOM + src + r"""
assert.ok(Math.abs(OPENING_SURROUND_FLOOR - 0.80) < 1e-12, "the shipping floor");
// THE FLOOR IS WHAT BOUNDS THE WHOLE TERM, and this is the assertion that
// keeps the docstring above honest. At 0.55 the surround was worth up to
// 1.82x, enough to buy an opening 8.6% worse by the page's own score -- and
// it did, on both viewports (BESTVIEW.md 6.1). At 0.80 it is worth at most
// 1.25x, so it separates frames that are close and cannot override one that
// is materially better.
assert.ok(1 / OPENING_SURROUND_FLOOR <= 1.25 + 1e-12,
          "the most the surround can ever be worth: " + (1 / OPENING_SURROUND_FLOOR));
// the term itself is bounded by the floor below and 1 above, whatever the pose
const term = i => OPENING_SURROUND_FLOOR + (1 - OPENING_SURROUND_FLOOR) * surroundOf(i);
for (let i = 0; i < cams.length; i++){
  const t = term(i);
  assert.ok(t >= OPENING_SURROUND_FLOOR - 1e-12 && t <= 1 + 1e-12, "bounded: " + t);
}
// a frame with nothing in it does not win on its surround
const good = {drawn: 0.99, distance: 6, detail: 0.13};   // a photograph
const poor = {drawn: 0.30, distance: 6, detail: 0.02};   // mostly void
assert.ok(openingScore(good, BLINKERED[3]) > openingScore(poor, ALLROUND[3]),
          "a factor, not a filter");
// between two frames that are the same, the surround decides -- which is the
// whole of what this term was added to do
assert.ok(openingScore(good, ALLROUND[3]) > openingScore(good, BLINKERED[3]),
          "and it does decide when the frames are level");
// and a caller that passes no index is scored exactly as before
assert.strictEqual(openingScore(good), openingScore(good, undefined));
assert.ok(openingScore(good) > openingScore(good, ALLROUND[3]) - 1e-12,
          "no index means no discount at all");
""")

    def test_the_scan_scores_every_pose_with_its_own_surround_and_reports_it(self):
        text = _template()
        choose = _section(text, "async function chooseOpening(", "function scorePoses(")
        assert "openingScore(r, i)" in choose, "the pose's own index, not the frame alone"
        assert "surround: +surroundOf(i).toFixed(3)" in choose
        # and the whole scanned table is published, so a review can check the
        # choice against its own measure rather than taking the page's word
        opening = _section(text, "  S.opening = {index: best", "return best;")
        for key in ("surround:", "surroundFloor:", "scanned:"):
            assert key in opening, key


class TestACameraInsideTheGeometryDoesNotStayThere:
    """`resistOnce` lets a step through whenever it does not make the bound
    WORSE. That is right at an edge and wrong inside one: inside the geometry
    `closeBound` is 1 everywhere, so nothing resists and nothing pushes back.
    The tube has had a release drift since the interaction lane; the standoff
    -- the bound a forward push actually meets -- had none.

    On the canonical world a grid scan of the tube finds a reachable point
    0.013 from the proxy, and on the unmodified page a camera released there
    is still 0.013 from it 150 frames later (BESTVIEW.md 6.6). This is that
    guard. It is NOT the fix for VISUAL-REVIEW-4 #3, whose camera is 3.84
    units from any proxy sample and outside the captured envelope."""

    def test_the_guard_targets_the_hard_limit_and_will_not_leave_the_tube(self):
        step = _section(_template(), "  function step(F, path, cam, ctl, dt, V){",
                        "  /* A uniformly random reachable view")
        assert "const d0 = nearest(F, out.p);" in step
        assert "if (d0 < D_MIN && d0 > 0 && isFinite(d0)){" in step, \
            "only inside the standoff, and never on a camera with no geometry near it"
        # D_MIN, not D_SOFT: it cannot move a camera that is merely close, and
        # over all 198 recorded poses the nearest any comes to the proxy is
        # 0.886 -- outside D_SOFT itself, so no pose the wearer stood at moves
        assert "(1 - Math.exp(-dt / 450)) * (D_MIN - d0)" in step
        assert "D_SOFT - d0" not in step
        # and it will never trade one violated bound for another
        assert "pathDist(path, to).e <= Math.max(1, pathDist(path, out.p).e)" in step

    def test_a_camera_inside_the_standoff_eases_out_to_it_and_stops(self):
        _run_nav(r"""
// one proxy sample right beside the walk, so a camera ON the path is inside
// the standoff -- which is the shape of the canonical world's worst point
const near = Float32Array.from([...F.input.samples, 2.5, 0, 0.4]);
const F4 = Object.assign({}, F, {input: Object.assign({}, F.input, {samples: near})});
const released = {look: [0, 0], move: [0, 0, 0], held: false};
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1};
const d0 = NAV.nearest(F4, cam.p);
assert.ok(d0 > 0 && d0 < NAV.D_MIN, "it starts inside the standoff: " + d0);
const trace = [];
let prev = d0, maxE = 0;
for (let i = 0; i < 400; i++){
  cam = NAV.step(F4, path, cam, released, 16, V);
  const d = NAV.nearest(F4, cam.p);
  assert.ok(d >= prev - 1e-9, "it only ever eases outward: " + prev + " -> " + d);
  assert.ok(d <= NAV.D_MIN + 1e-6, "and never overshoots the limit: " + d);
  maxE = Math.max(maxE, NAV.pathDist(path, cam.p).e);
  prev = d;
  if (i % 50 === 0) trace.push(+d.toFixed(3));
}
assert.ok(prev > NAV.D_MIN - 0.02,
          "it reaches the standoff, it does not crawl: " + d0 + " -> " + prev + " " + trace);
assert.ok(maxE <= 1 + 1e-9, "and it never left the tube to do it: " + maxE);
""")

    def test_a_camera_that_is_merely_close_is_left_exactly_where_it_is(self):
        """The guard must not be a second envelope. It fires inside D_MIN and
        nowhere else, so a reader standing near the desk -- or a recorded pose
        the wearer actually stood at -- is not quietly pushed off it."""
        _run_nav(r"""
// 0.75 away: inside D_SOFT (0.9), outside D_MIN (0.6). Nothing should move.
const near = Float32Array.from([...F.input.samples, 2.5, 0, 0.75]);
const F4 = Object.assign({}, F, {input: Object.assign({}, F.input, {samples: near})});
const released = {look: [0, 0], move: [0, 0, 0], held: false};
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1};
const d0 = NAV.nearest(F4, cam.p);
assert.ok(d0 > NAV.D_MIN && d0 < NAV.D_SOFT, "merely close: " + d0);
const p0 = cam.p.slice();
for (let i = 0; i < 400; i++) cam = NAV.step(F4, path, cam, released, 16, V);
assert.ok(dist(cam.p, p0) < 1e-9,
          "a camera outside the standoff is not moved by it: " + dist(cam.p, p0));
""")


# NAV_ROOM at any scale `S`, under a NAV built for a given `OPT` (`mkNAV`).
NAV_ROOM_AT = r"""
function room(NAV, S){
  const up = [0, 1, 0], cams = [];
  for (let i = 0; i <= 10; i++) cams.push([i * 0.5 * S, 0, 0, 0, 0, 1]);
  const path = NAV.makePath(cams, up);
  const samples = [], off = [0], idx = [];
  for (let x = -4; x <= 9; x += 0.1) for (let y = -2.5; y <= 2.5; y += 0.1){
    samples.push(x * S, y * S, 3 * S);
    for (let k = 0; k < cams.length; k++) idx.push(k);
    off.push(idx.length);
  }
  const centres = Float32Array.from(cams.flatMap(c => [c[0], c[1], c[2]]));
  const F = NAV.fieldJob({samples: Float32Array.from(samples), seenOff: Int32Array.from(off),
                          seenIdx: Int32Array.from(idx), centres, path});
  while (!NAV.fieldWork(F, 50)) {}
  const dir = (yaw, pitch) => [Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), Math.cos(yaw) * Math.cos(pitch)];
  const V = {dir, up, fy: 1.25, aspect: 1.2};
  const sup = (p, yaw) => NAV.support(F, p, dir(yaw, 0), up, V.fy, V.aspect).s;
  return {NAV, S, F, path, V, sup, at: [2.5 * S, 0, 0]};
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const released = {look: [0, 0], move: [0, 0, 0], held: false};
"""


class TestASmallRoomIsTheSameRoomSmaller:
    """Walk 1 (`ee48aae3`, 2026-09-24): "it keeps snapping me to a view from the
    top-left of the desk, says nothing was photographed this way, and snaps back
    whenever I move". A solve's gauge has no scale, and that room came out with a
    median scene depth of 0.174 against the 4.7 every length on the page was
    measured on: its whole proxy (0.47 x 0.51 x 0.46) sat inside the 0.6 standoff,
    so every released camera was eased 0.6 out of the room -- all 53 recorded
    phone poses, 3.4 scene depths each. Every length is now a multiple of the
    scene's own unit (`OPT.sceneUnit`, median scene depth / 4.7)."""

    U = 0.1741 / 4.7

    def _run(self, script):
        import subprocess

        program = ("const assert = require('assert');\n"
                   "const mkNAV = OPT => {\n" + _nav_source() + "\nreturn NAV;\n};\n"
                   + NAV_ROOM_AT + f"\nconst u = {self.U!r};\n" + script
                   + "\nconsole.log('nav ok');\n")
        r = subprocess.run([_node(), "-"], input=program, capture_output=True, text=True,
                           timeout=120)
        assert r.returncode == 0 and "nav ok" in r.stdout, (r.stdout + r.stderr)[-3000:]

    def test_the_small_room_moves_exactly_as_the_reference_room_scaled(self):
        self._run(r"""
const ref = room(mkNAV({standoff: 0.6, sceneUnit: 1}), 1);
const small = room(mkNAV({standoff: 0.6, sceneUnit: u}), u);
assert.ok(Math.abs(small.NAV.D_MIN - 0.6 * u) < 1e-12 && Math.abs(small.NAV.R_MAX - 2.2 * u) < 1e-12);
// the same support field, point for point (to rounding: this room's samples sit
// on a regular grid, so some fall exactly on a direction cell's edge)
for (const [p, yaw] of [[[2.5, 0, 0], 0], [[2.5, 0, 0], 1.2], [[1.1, 0.3, -0.5], 0.4], [[4.2, -0.2, 1.3], -0.3]]){
  const a = ref.sup(p, yaw), b = small.sup(p.map(x => x * u), yaw);
  assert.ok(Math.abs(a - b) < 0.02, "support " + p + " @" + yaw + ": " + a + " vs " + b);
}
assert.ok(small.sup(small.at, 0) > 0.95 && small.sup(small.at, Math.PI) === 0);
// the same hands make the same walk: a push into the wall (the standoff and the
// tube both stop it), then released (the drift back to the soft edge)
function walk(R){
  let cam = {p: R.at.slice(), yaw: 0, pitch: 0, floor: 1};
  const out = [];
  for (let i = 0; i < 120; i++)
    cam = R.NAV.step(R.F, R.path, cam, {look: [0, 0], move: [0, 0, 0.05 * R.S], held: true}, 16, R.V);
  out.push(cam.p.map(x => x / R.S));
  for (let i = 0; i < 300; i++){ cam = R.NAV.step(R.F, R.path, cam, released, 16, R.V); out.push(cam.p.map(x => x / R.S)); }
  return out;
}
const a = walk(ref), b = walk(small);
assert.ok(a[0][2] > 1.5, "the push went somewhere: " + a[0]);
for (let i = 0; i < a.length; i++) assert.ok(dist(a[i], b[i]) < 1e-4, "frame " + i + ": " + a[i] + " vs " + b[i]);
// and a recorded pose, 3 scene units from the wall it faced, is where the camera stays
let cam = {p: small.at.slice(), yaw: 0, pitch: 0, floor: 1, aimed: true};
for (let i = 0; i < 300; i++) cam = small.NAV.step(small.F, small.path, cam, released, 16, small.V);
assert.ok(dist(cam.p, small.at) < 1e-12, "the wearer's pose is left alone: " + dist(cam.p, small.at));
assert.ok(cam.dark < 0.1, "and facing what they photographed is not called dark: " + cam.dark);
""")

    def test_without_the_unit_the_small_room_throws_the_camera_out(self):
        """The mechanism, pinned: the page as it was at 4b4b444 (no unit)."""
        self._run(r"""
const blind = room(mkNAV({standoff: 0.6, sceneUnit: 1}), u);
let cam = {p: blind.at.slice(), yaw: 0, pitch: 0, floor: 1, aimed: true};
for (let i = 0; i < 300; i++) cam = blind.NAV.step(blind.F, blind.path, cam, released, 16, blind.V);
assert.ok(dist(cam.p, blind.at) > 3 * u, "eased out past the wall's own distance: " + dist(cam.p, blind.at));
""")

    def test_the_page_takes_its_unit_from_the_scene_it_serves(self):
        text = _template().replace("\r\n", "\n")
        assert ('sceneUnit: (u => u > 0 && isFinite(u) ? Math.min(1e3, Math.max(1e-3, u)) : 1)'
                '(+(Q.get("su") ?? (CONFIG.median_scene_depth || 4.7) / 4.7)),') in text
        assert "const UNIT = (typeof OPT === \"object\" && OPT && OPT.sceneUnit > 0) ? OPT.sceneUnit : 1;" in text
        # the lengths outside NAV: a finger's travel, the hole reach, the source
        # tests, the near plane, the Best view's own distances
        for length in ("const MOVE_PER_PX = 0.004 * OPT.sceneUnit, PINCH_UNITS = 2.0 * OPT.sceneUnit;",
                       "const KEY_SPEED = 1.1 / 1000 * OPT.sceneUnit;",
                       "ctl.move[2] += -e.deltaY * 0.0025 * OPT.sceneUnit;",
                       "const HOLE_PERIMETER = 0.8 * OPT.sceneUnit, HOLE_REACH = 0.35 * OPT.sceneUnit;",
                       "return z <= zs * 1.03 + 0.03 * OPT.sceneUnit;",
                       "near: Math.max(0.02 * OPT.sceneUnit, diag * 0.0015)",
                       "if (rise < 0.15 * OPT.sceneUnit) continue;"):
            assert length in text, length
        assert text.count("if (z < 0.05 * OPT.sceneUnit) continue;") == 2
