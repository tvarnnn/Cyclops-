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
        assert "faces redacted" in html and "nothing there is filled in" in html
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
        blend = text[text.index("const FS_BLEND"):text.index("/* ---------- main")]
        for lighting in ("dot(n,", "normalize(cross(dFdx", "hemi", "uUpView"):
            assert lighting not in blend

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
        blend = _section(_template(), "const FS_BLEND", "/* ---------- main")
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
        blend = _section(_template(), "const FS_BLEND", "/* ---------- main")
        assert "uConsensus" in blend and "float cdist(" in blend
        # the voters are the k + 2 best, never every candidate
        assert "vthr = (n > k + 2) ? s[k + 2]" in blend

    def test_the_photometric_model_is_applied(self):
        text = _template()
        blend = _section(text, "const FS_BLEND", "/* ---------- main")
        assert "uGain[i] * exp(dot(uSlope[i], q) + uVig.x * r2 + uVig.y * r2 * r2)" in blend
        assert "kf.gain_slope" in text and "man.exposure.vignette" in text

    def test_highlights_roll_off_below_white(self, built):
        from tower.world_builder.appearance_render import build_appearance_config

        text = _template()
        blend = _section(text, "const FS_BLEND", "/* ---------- main")
        ceiling = float(re.search(r"#define TONE_CEILING ([0-9.]+)", blend).group(1))
        assert ceiling < 0.99                                   # the brightest output is not white
        assert "float y = uKnee + span * (1.0 - exp(-(m - uKnee) / span));" in blend
        assert "if (m <= uKnee) return x;" in blend
        knee = build_appearance_config(built.store, WORLD, SESSION)["tone_knee"]
        assert 0.3 <= knee < ceiling

    def test_a_changed_choice_of_sources_crossfades(self):
        text = _template()
        assert "function setChosen(" in text and "function updatePresence(" in text
        assert "uPres[i]" in _section(text, "const FS_BLEND", "/* ---------- main")
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
        text = _template()
        blend = _section(text, "const FS_BLEND", "/* ---------- main")
        assert "if (!observed){ o = vec4(0.0); return; }" in blend      # no tint, no fog
        assert "uFogLift" not in text and "uGhost" not in text
        # the depth prepass still draws the whole proxy before the blend
        pass_ = _section(text, "function blendPass(", "const P = G.blend;")
        assert "gl.colorMask(false, false, false, false);" in pass_ and "drawMesh();" in pass_

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
        assert "restoreContext()" in lost and "location.reload()" in lost


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
