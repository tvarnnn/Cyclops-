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
        assert len(html.encode("utf-8")) < 256 * 1024, "a shell, not a data page"
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
        assert "A grey haze is a place no kept frame saw" in html
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
        assert "if (!observed){ o = vec4(0.0); return; }" in blend      # no tint, no fog
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
let cam = {p: [2.5, 0, -0.95], yaw: 0, pitch: 0, floor: 1};
let e = NAV.pathDist(path, cam.p).e;
for (let i = 0; i < 400; i++){
  const n = NAV.step(F, path, cam, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
  assert.ok(dist(n.p, cam.p) < 0.02, "a drift, never a snap");
  const e1 = NAV.pathDist(path, n.p).e;
  assert.ok(e1 <= e + 1e-9);
  e = e1; cam = n;
}
assert.ok(e < 0.65 && e > 0.5, "back to the soft edge, not dragged to the walk: " + e);
// a held camera is not moved by the drift
const held = NAV.step(F, path, {p: [2.5, 0, -0.95], yaw: 0, pitch: 0, floor: 1}, {look: [0, 0], move: [0, 0, 0], held: true}, 16, V);
assert.deepStrictEqual(held.p, [2.5, 0, -0.95]);
""")

    def test_looking_toward_what_was_never_captured_is_resisted(self):
        _run_nav(r"""
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1}, least = 1, resisted = 0;
for (let i = 0; i < 400; i++){
  cam = NAV.step(F, path, cam, {look: [0.03, 0], move: [0, 0, 0], held: true}, 16, V);
  least = Math.min(least, sup(cam.p, cam.yaw)); resisted = Math.max(resisted, cam.resisted);
}
assert.ok(least >= NAV.T_LO - 0.02, "the look stops before the view is mostly dark: " + least);
assert.ok(cam.yaw > 0.5, "but it turns freely while the view is supported: " + cam.yaw);
assert.ok(resisted > 0.35);
for (let i = 0; i < 400; i++) cam = NAV.step(F, path, cam, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.ok(sup(cam.p, cam.yaw) >= NAV.T_HI - 0.05, "released, it eases back toward what was seen");
// turning back toward the wall is not held back (the field is sampled on a
// cube map, so support is only monotone to within a cell)
const back = NAV.step(F, path, cam, {look: [-0.03, 0], move: [0, 0, 0], held: true}, 16, V);
assert.ok(cam.yaw - back.yaw > 0.027 && back.resisted < 0.1, (cam.yaw - back.yaw) + " " + back.resisted);
""")

    def test_a_recorded_view_that_is_itself_dark_is_never_pushed(self):
        _run_nav(r"""
// the wearer looked away from the wall: support 0, but it is a recorded pose
const pose = {p: at.slice(), yaw: Math.PI, pitch: 0, floor: 0.05};
const n = NAV.step(F, path, pose, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.strictEqual(n.yaw, Math.PI); assert.deepStrictEqual(n.p, at);
// and without a field (still building) the camera does not move at all:
// early input used to escape the envelope (visual review, item 5)
const wait = NAV.step(null, path, {p: at.slice(), yaw: 0, pitch: 0}, {look: [2, 0], move: [0, 0, -5], held: true}, 16, V);
assert.strictEqual(wait.yaw, 0); assert.deepStrictEqual(wait.p, at); assert.strictEqual(wait.waiting, true);
""")

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
assert.ok(NAV.glideMs(a, b) >= 400 && NAV.glideMs(a, {p: [30, 0, 0], yaw: 3, pitch: 0}) <= 1600);
// a gap in the recorded walk longer than JUMP is not a corridor
const gap = NAV.makePath([[0, 0, 0, 0, 0, 1], [NAV.JUMP + 3, 0, 0, 0, 0, 1]], up);
assert.ok(NAV.pathDist(gap, [(NAV.JUMP + 3) / 2, 0, 0]).e > 1);
""")

    def test_reachable_samples_are_inside_and_supported(self):
        _run_nav(r"""
let seed = 7; const rand = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
for (let i = 0; i < 40; i++){
  const r = NAV.sampleReachable(F, path, V, rand);
  assert.ok(r, "found one");
  assert.ok(NAV.pathDist(path, r.p).e <= 1 && r.support >= NAV.T_LO);
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
        assert config["standoff"] == R.STANDOFF == 1.0
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
        start = _section(text, "async function finishOpening(", "/* -------- start ---")
        assert start.index("setPose(poseOf(ci));") < start.index("shownAt =") < start.index("frame(true);")
        assert "uShow" in _section(text, "const FS_COMPOSITE", "}`;")


class TestTheEnvelopeStopsBeforeTheUglyFrame:

    def test_the_floor_after_the_build_is_the_recorded_poses(self):
        text = _template()
        build = _section(text, "function buildNav(){", "/* Overview:")
        assert "cam.floor = floorAt(poseOf(ci));" in build and "cam.floor = floorAt(cam);" not in build
        update = _section(text, "function navUpdate(", "function mulberry(")
        assert "if (next.waiting){ vel.look = [0, 0]; vel.move = [0, 0, 0];" in update

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

    def test_a_look_held_against_the_edge_crosses_to_the_next_supported_direction(self):
        _run_nav(r"""
// at the wall-facing spot, turning right past the edge: the field has support
// only toward +z, so from yaw -pi/2 pushing further there is nothing within
// half a turn except back through the wall
const y = NAV.lookAcross(F, at, Math.PI, 0, 1, V);
assert.ok(y !== null, "found the wall on the far side");
assert.ok(NAV.support(F, at, dir(y, 0), up, V.fy, V.aspect).s >= NAV.T_HI);
assert.ok(y - Math.PI > 0.3 && y - Math.PI <= Math.PI + 1e-9);
assert.strictEqual(NAV.lookAcross(null, at, 0, 0, 1, V), null);
""")
        text = _template()
        update = _section(text, "function navUpdate(", "function mulberry(")
        assert "NAV.lookAcross(navField, cam.p, cam.yaw, cam.pitch, Math.sign(lookYaw), V)" in update
        assert "glideTo({p: cam.p.slice(), yaw: y, pitch: cam.pitch}, null);" in update

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
        assert "NAV.nextPose(poseQ, ci, d, POSE_MIN, poseC, POSE_CONTENT)" in step
        assert "scorePoses();" in _section(text, "function buildNav(){", "/* Overview:")

    def test_overview_is_a_distinct_vantage(self):
        over = _section(_template(), "const OVERVIEW_AWAY", "function overview(){")
        assert "< OVERVIEW_AWAY) continue;" in over
        assert "r.dist < OVERVIEW_DEPTH * ref" in over and "0.4 + 0.6 * Math.min(1, r.distance / (1.5 * ref))" in over

    def test_the_look_stays_within_the_pitch_the_walk_looked_at(self):
        _run_nav(r"""
const VP = Object.assign({}, V, {pitchMin: -0.4, pitchMax: 0.3});
let cam = {p: at.slice(), yaw: 0, pitch: 0, floor: 1}, prev = 0, resisted = 0;
for (let i = 0; i < 300; i++){
  const n = NAV.step(F, path, cam, {look: [0, 0.02], move: [0, 0, 0], held: true}, 16, VP);
  assert.ok(n.pitch <= 0.3 + 1e-9, "never above the recorded range: " + n.pitch);
  if (n.pitch > 0.3 - NAV.PITCH_SOFT) assert.ok(n.pitch - cam.pitch <= prev + 1e-9, "slowing, not a wall");
  prev = n.pitch - cam.pitch; resisted = Math.max(resisted, n.resisted); cam = n;
}
assert.ok(cam.pitch > 0.2 && resisted > 0.35, cam.pitch + " " + resisted);
for (let i = 0; i < 300; i++) cam = NAV.step(F, path, cam, {look: [0, -0.02], move: [0, 0, 0], held: true}, 16, VP);
assert.ok(cam.pitch >= -0.4 - 1e-9 && cam.pitch < -0.3, "and not below it: " + cam.pitch);
""")
        text = _template()
        assert "pitchMin: pitchRange[0], pitchMax: pitchRange[1]" in text and "recordedPitch();" in text

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
    margin; the envelope knows a drawn-but-empty view from a room view, and
    nudges away from it without ever refusing a deliberate look."""

    def test_the_field_carries_content_beside_coverage(self):
        _run_nav(NAV_CONTENT + r"""
// the same wall, equally well covered from both halves
assert.ok(Math.abs(supC(at, 0.6) - supC(at, -0.6)) < 0.08, "coverage does not tell them apart");
assert.ok(con(at, 0.6) > 0.7, "the textured half: " + con(at, 0.6));
assert.ok(con(at, -0.6) < 0.15, "the blank half: " + con(at, -0.6));
// richness is the 0..1 reading of it, and monotone between the thresholds
assert.strictEqual(NAV.richness(NAV.C_LO), 0);
assert.strictEqual(NAV.richness(NAV.C_HI), 1);
let last = -1;
for (let c = 0; c <= 0.5; c += 0.01){ const r = NAV.richness(c); assert.ok(r >= last - 1e-12); last = r; }
// a field built without content says nothing rather than something wrong
assert.strictEqual(NAV.support(F, at, dir(0, 0), up, V.fy, V.aspect).c, 0);
""")

    def test_a_featureless_view_is_a_nudge_and_never_a_wall(self):
        _run_nav(NAV_CONTENT + r"""
// the bound itself can never reach the value at which a step is refused
assert.strictEqual(NAV.contentBound(0), NAV.C_BOUND_MAX);
assert.ok(NAV.C_BOUND_MAX < 0.999, "a content bound must never refuse a step");
assert.strictEqual(NAV.contentBound(1), 0);
// Looking from the textured half to the blank half: slower, but it gets there.
// (Past yaw -0.85 this synthetic wall runs out and the SUPPORT limit takes
// over, a different mechanism, so the content nudge is measured inside the
// supported range.)
let cam = {p: at.slice(), yaw: 0.6, pitch: 0, floor: 1}, worst = 1, steps = 0;
while (cam.yaw > -0.7 && steps < 400){
  const n = NAV.step(FC, path, cam, {look: [-0.01, 0], move: [0, 0, 0], held: true}, 16, V);
  worst = Math.min(worst, Math.abs(n.yaw - cam.yaw) / 0.01);
  cam = n; steps++;
}
assert.ok(cam.yaw <= -0.7, "the deliberate look reached the blank wall: " + cam.yaw);
assert.ok(steps >= 130, "it never went faster than asked: " + steps);
assert.ok(worst < 0.95, "it was resisted on the way: " + worst);
assert.ok(worst > 0.2, "but never stopped: " + worst);
""")

    def test_released_on_nothing_the_look_eases_toward_content_and_stops(self):
        _run_nav(NAV_CONTENT + r"""
// parked on the blank half with nothing held: the look eases toward the texture
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
// and a look of the person's own gives the budget back
const after = NAV.step(FC, path, cam, {look: [-0.05, 0], move: [0, 0, 0], held: true}, 16, V);
assert.strictEqual(after.cdrift, 0);
// on a view that already has content in it there is no ease at all
let rich = {p: at.slice(), yaw: 0.6, pitch: 0, floor: 1};
for (let i = 0; i < 300; i++) rich = NAV.step(FC, path, rich, {look: [0, 0], move: [0, 0, 0], held: false}, 16, V);
assert.strictEqual(rich.yaw, 0.6, "a room view is left alone");
""")

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
  assert.ok(r.support >= NAV.T_LO);
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
        step = _section(text, "  function step(F, path, cam, ctl, dt, V){", "  /* Looking ACROSS")
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

    def test_the_edge_hint_is_for_the_edge_and_not_for_a_dull_view(self):
        text = _template()
        update = _section(text, "function navUpdate(", "function mulberry(")
        assert "hint(next.hard || 0);" in update
        step = _section(text, "  function step(F, path, cam, ctl, dt, V){", "  /* Looking ACROSS")
        assert "const h = lookBound(r.s, out.floor); hard = h;" in step


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
        for step in ("opening = await chooseOpening();", "setPose(poseOf(ci));",
                     "shownAt =", 'S.phase = "ready";', "buildNav();"):
            assert step in opening, step


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
        assert "opening = await chooseOpening();" in text

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
