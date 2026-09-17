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

        html = build_world_render(built.store, WORLD, SESSION)
        assert _meta(html, "wb-representation") == "appearance"
        assert "Captured images on reconstructed geometry" in html

    def test_the_page_and_the_revision_route_agree(self, built):
        from tower.results.world_builder_render import build_render_revision, build_world_render

        rev = build_render_revision(built.store, WORLD, SESSION)
        assert rev["representation"] == "appearance"
        assert rev["revision"] == f"{SESSION}/appearance:1"
        html = build_world_render(built.store, WORLD, SESSION)
        assert _meta(html, "wb-revision") == rev["revision"]
        assert _config(html)["appearance_revision"] == rev["appearance"]["revision"]

    def test_a_new_appearance_build_does_not_move_the_page_revision(self, built):
        """The page follows builds itself; a moved page revision would make the
        phone reload it and reset the wearer's camera on every solve."""
        from tower.results.world_builder_render import build_render_revision

        first = build_render_revision(built.store, WORLD, SESSION)
        built.build(force=True, redactor_factory=_never_redact)
        second = build_render_revision(built.store, WORLD, SESSION)
        assert second["appearance"]["revision"] != first["appearance"]["revision"]
        assert second["revision"] == first["revision"]
        assert second["representation"] == "appearance"

    def test_an_appearance_behind_the_surface_is_still_the_top_rung(self, built):
        """Currency is reported, never enforced: a live walk's new surface must
        not swap the page down and back up for the minute a rebuild takes."""
        from tower.results.world_builder_render import build_render_revision, build_world_render

        built.write_surface(subdivisions=12)
        rev = build_render_revision(built.store, WORLD, SESSION)
        assert rev["representation"] == "appearance"
        assert rev["appearance"]["current"] is False
        config = _config(build_world_render(built.store, WORLD, SESSION))
        assert config["current"] is False and config["currency_reason"]

    def test_a_relabelled_session_steps_down_to_the_surface(self, built):
        from tower.results.world_builder_render import (
            WorldRenderUnavailable,
            build_render_revision,
            build_world_render,
        )

        built.set_label("faces-detected-and-filled/yunet-2023mar@0.30+plausibility1")
        html = build_world_render(built.store, WORLD, SESSION)
        assert _meta(html, "wb-representation") == "surface"
        assert build_render_revision(built.store, WORLD, SESSION)["representation"] == "surface"
        with pytest.raises(WorldRenderUnavailable):
            build_world_render(built.store, WORLD, SESSION, representation="appearance")

    def test_a_purged_world_steps_down_too(self, built):
        from tower.results.world_builder_render import build_world_render

        record = built.store.read_world(WORLD)
        built.store.write_world(type(record)(**{**record.__dict__, "images_purged": True}))
        assert _meta(build_world_render(built.store, WORLD, SESSION), "wb-representation") != "appearance"

    def test_a_world_without_appearance_gets_the_page_it_got_before(self, tmp_path):
        from tower.results.world_builder_render import build_world_render

        world = World(tmp_path)
        auto = build_world_render(world.store, WORLD, SESSION)
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
            build_world_render(built.store, WORLD, SESSION, view="diagnostics")
        assert build_render_revision(built.store, WORLD, SESSION,
                                     view="diagnostics")["representation"] == "sparse"

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
        html = build_world_render(built.store, WORLD, SESSION)
        assert _meta(html, "wb-representation") == "surface"

    def test_a_page_that_cannot_be_composed_falls_back(self, built, monkeypatch):
        from tower.results.world_builder_render import build_world_render
        from tower.world_builder import appearance_render as AR

        def refuse(*a, **kw):
            raise AR.AppearanceViewerUnavailable("simulated")

        monkeypatch.setattr(AR, "build_appearance_page", refuse)
        assert _meta(build_world_render(built.store, WORLD, SESSION), "wb-representation") == "surface"


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


class TestTheRoute:

    def test_the_header_policy_matches_the_page(self, built):
        client = _client(built)
        app = client.get(f"/worlds/{WORLD}/render")
        assert app.status_code == 200
        assert app.headers["content-security-policy"] == APP_POLICY == _meta_policy(app.text)
        assert app.headers["cache-control"] == "no-store"
        debug = client.get(f"/worlds/{WORLD}/render", params={"transport": "tower"})
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
        rev = client.get(f"/worlds/{WORLD}/render/revision").json()
        assert rev["representation"] == "appearance"
        assert len(json.dumps(rev)) < 512
