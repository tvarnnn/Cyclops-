"""A Tower must not offer Scene Understanding it cannot run.

`build_live_cartridges` has always try/excepted `_scene_session`, and
`main.py` has always derived `scene_enabled` from whether that returned a
session -- so the honest mechanism was there. It simply never fired for a
missing torch, because `_resolve_device` imports torch ONLY when the
device is not "cpu", and "cpu" is the default. On a host without the
`[ml]` extra the session constructed happily, `/cartridges` said
`available: true`, and `start()` then failed in 51 ms with a
ModuleNotFoundError. The declaration never self-corrected, because it is
a pure function of configuration.

The fix is an eager `import torch, torchvision` in `_scene_session`, so
the existing mechanism sees what it was always meant to see.

`find_spec` was measured as the alternative and refused: it locates
without executing, so it reported `available: true` for a package whose
loader raises, and answered `POST /scene/start` with 200.

Absence is simulated with a `sys.meta_path` blocker rather than by
uninstalling anything, so these run on a host that HAS torch -- which is
the host this suite normally runs on, and therefore the only way this
would ever be exercised.
"""

import sys

import pytest

from tower.results import registry


class _BlockModules:
    """Make named top-level modules un-importable, and nothing else."""

    def __init__(self, *names):
        self._names = set(names)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in self._names:
            # `name=` is not decoration. CPython's import system always
            # sets it, and `client_safe_reason` READS it -- it reports the
            # module name and discards the message, because CPython puts
            # the module's `__file__` in an ImportError's text and that
            # string reaches an unauthenticated wire. A blocker that
            # omitted `name` simulated an exception the import system
            # cannot produce, and the test that noticed was asserting the
            # reason names the missing module.
            raise ModuleNotFoundError(f"No module named {root!r}", name=root)
        return None

    def __enter__(self):
        self._saved = {
            name: module
            for name, module in sys.modules.items()
            if name.split(".")[0] in self._names
        }
        for name in list(self._saved):
            del sys.modules[name]
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        sys.meta_path.remove(self)
        sys.modules.update(self._saved)
        return False


@pytest.fixture
def settings_scene_on(monkeypatch):
    """Scene on, device `cpu` -- the DEFAULT, and the only broken config.

    Built through `get_settings()` rather than by hand, following
    `test_live_cartridge_privacy.py`, so the env-var path this is really
    about is the one under test. `cpu` is stated explicitly because it is
    load-bearing: on `auto`/`cuda`, `_resolve_device` already imported
    torch and the declaration was already correct.
    """
    from tower.config import get_settings

    monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "true")
    monkeypatch.setenv("TOWER_SCENE_DEVICE", "cpu")
    monkeypatch.delenv("TOWER_SCENE_ORIENTATION", raising=False)
    return get_settings()


class TestATorchlessHostDoesNotOfferScene:
    def test_the_session_is_not_constructed_without_torch(
        self, settings_scene_on
    ):
        """RED before the eager import: a session was returned happily."""
        from tower.cartridge_runtime import build_live_cartridges

        with _BlockModules("torch", "torchvision"):
            live = build_live_cartridges(settings_scene_on)

        assert live.scene is None
        assert live.frame_consumers == []

    def test_the_declaration_reports_it_unavailable(self, settings_scene_on):
        """RED before: `/cartridges` said available: true on a dead cartridge."""
        from tower.cartridge_runtime import build_live_cartridges

        with _BlockModules("torch", "torchvision"):
            live = build_live_cartridges(settings_scene_on)

        declaration = registry.declare(
            None,
            scene_enabled=live.scene is not None,
            scene_unavailable_reason=live.scene_unavailable_reason,
        )
        offer = next(
            entry
            for entry in declaration["cartridges"]
            if entry["cartridge"] == "scene_understanding"
        )

        assert offer["available"] is False
        # Still OFFERED. "This build implements the contract" is the third
        # state -- update-the-app and cannot-run-it-here are different
        # answers and iOS renders them differently.
        assert offer["contract"] == "scene_understanding.live/2026-08-27"

    def test_a_torch_present_host_is_completely_unaffected(
        self, settings_scene_on
    ):
        """The production host. If this ever fails the fix is not free."""
        from tower.cartridge_runtime import build_live_cartridges

        live = build_live_cartridges(settings_scene_on)

        assert live.scene is not None
        assert live.scene_unavailable_reason is None


class TestTheReasonNamesWhatIsActuallyWrong:
    """The declaration used to blame a variable that was already set.

    With the flag ON and construction failing, `/cartridges` served
    SCENE_DISABLED_REASON -- "TOWER_SCENE_UNDERSTANDING is unset or off" --
    which sends an operator to check the one thing that is already
    correct. Reachable on the tree before this change via
    `TOWER_SCENE_DEVICE=cuda` with torch absent; the eager import would
    have made it reachable in the DEFAULT configuration, so the reason had
    to travel with it.
    """

    def test_it_does_not_claim_the_variable_is_unset(self, settings_scene_on):
        from tower.cartridge_runtime import build_live_cartridges

        with _BlockModules("torch", "torchvision"):
            live = build_live_cartridges(settings_scene_on)

        reason = live.scene_unavailable_reason
        assert reason is not None
        assert "unset or off" not in reason
        assert "torch" in reason

    def test_configured_off_keeps_the_pinned_wording(self, monkeypatch):
        """Switched off must say so. Only a FAILURE replaces that sentence.

        Set explicitly to "off": since 2026-09-07 an UNSET variable means
        auto, and on this host auto offers the cartridge.
        """
        from tower.config import get_settings
        from tower.cartridge_runtime import build_live_cartridges

        monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "off")
        live = build_live_cartridges(get_settings())
        declaration = registry.declare(
            None,
            scene_enabled=live.scene is not None,
            scene_unavailable_reason=live.scene_unavailable_reason,
        )
        offer = next(
            entry
            for entry in declaration["cartridges"]
            if entry["cartridge"] == "scene_understanding"
        )

        assert offer["unavailable_reason"] == registry.SCENE_DISABLED_REASON

    def test_the_reason_cannot_disclose_a_filesystem_path(self):
        """This string reaches an unauthenticated `/cartridges`.

        `client_safe_reason` reduces an OSError to its type name because
        an OSError describes a failure by naming the path it happened on,
        and that path discloses the home directory and the OS username.
        """
        from tower.logging_config import client_safe_reason

        leaked = FileNotFoundError(
            2, "No such file or directory", r"C:\Users\someone\weights.onnx"
        )

        assert client_safe_reason(leaked) == "FileNotFoundError"


class TestEverySceneSurfaceAgrees:
    """Drive the REAL app, not hand-set `app.state`.

    The reason travels four hops -- `LiveCartridges` -> `main.py` ->
    `declaration_inputs` -> `declare()`/`find_offer` -> the routes -- and
    every other test here sets `app.state` by hand or calls `declare()`
    with the keyword directly. That meant the thread was never pinned end
    to end, and two independent reviewers found the same consequence: one
    hop was missed. `find_offer` had not been taught the keyword, so the
    WebSocket subscribe refusal went on naming the environment variable
    while three other surfaces told the truth -- and it is the surface a
    phone actually receives.

    This builds the app under a torch blocker and asks every surface.
    """

    def _app_without_torch(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", "true")
        monkeypatch.setenv("TOWER_SCENE_DEVICE", "cpu")
        monkeypatch.delenv("TOWER_CAPTURE_ROOT", raising=False)
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)
        with _BlockModules("torch", "torchvision"):
            from tower.main import create_app

            return create_app()

    def test_no_surface_blames_the_variable_that_is_already_set(
        self, monkeypatch, tmp_path
    ):
        from fastapi.testclient import TestClient

        app = self._app_without_torch(monkeypatch, tmp_path)
        assert app.state.scene_enabled is False
        reason = app.state.scene_unavailable_reason
        assert reason and "unset or off" not in reason

        client = TestClient(app)

        http = next(
            entry
            for entry in client.get("/cartridges").json()["cartridges"]
            if entry["cartridge"] == "scene_understanding"
        )
        assert http["available"] is False
        assert http["unavailable_reason"] == reason

        assert client.get("/scene").json()["detail"] == reason

        # The socket subscribe refusal -- the surface a phone receives,
        # and the one that was missed.
        offer = registry.find_offer(
            None, "scene_understanding", "live",
            **{
                key: value
                for key, value in registry.declaration_inputs(app.state).items()
                if key != "world_root"
            },
        )
        assert offer["available"] is False
        assert offer["unavailable_reason"] == reason


class TestTheContractSaysSo:
    """The wire changed meaning, so the prose had to change with it.

    No FIELD moved -- `available` and `unavailable_reason` already
    existed, and `ios/scripts/contract-drift-check.py` reads neither -- but
    a Tower that used to answer `true` on a torch-less host now answers
    `false`, and a contract that still said availability was purely about
    configuration would be describing a Tower that no longer exists.

    Path derived from `__file__` rather than the cwd: this contract lives
    at the REPOSITORY root, one level above the `tower/` package that the
    rest of this suite resolves against, and this lane has already been
    bitten three times by cwd-relative resolution.
    """

    def _contract(self):
        """The contract with whitespace normalised.

        Markdown wraps at ~72 columns, so a phrase these tests care about
        routinely straddles a newline. Asserting against the raw text
        makes the test fail when someone rewraps a paragraph, which is a
        false alarm that teaches people to delete tests.
        """
        import pathlib
        import re

        path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "docs" / "contracts" / "TOWER-UNIFIED-CARTRIDGES.md"
        )
        if not path.exists():  # pragma: no cover - layout guard
            pytest.skip(f"contract not found at {path}")
        return re.sub(r"\s+", " ", path.read_text(encoding="utf-8"))

    def test_it_no_longer_claims_availability_is_only_about_configuration(self):
        contract = self._contract()

        assert "CONFIGURATION, never about current activity" not in contract
        assert "never about current activity" in contract

    def test_it_states_that_a_missing_runtime_dependency_makes_it_unavailable(
        self,
    ):
        contract = self._contract()

        assert "[ml]" in contract
        assert "torch" in contract

    def test_it_no_longer_promises_the_reason_always_names_a_variable(self):
        """A missing extra has no `TOWER_` variable that would fix it."""
        contract = self._contract()

        offending = (
            "A `cartridge_unavailable` message names the `TOWER_` variable "
            "that would fix it."
        )
        assert offending not in contract
        # The qualified form survives, so the common case still reads the
        # way it always did.
        assert "where a variable is what is missing" in contract


class TestTheTwoSurfacesAgree:
    """`/cartridges` and `/scene/*` must not explain one configuration twice.

    That is why `SCENE_DISABLED_REASON` is a module constant rather than an
    inline string, and the specific reason has to follow the same rule.
    """

    def test_the_route_serves_the_same_reason_as_the_declaration(
        self, settings_scene_on
    ):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from tower.cartridge_runtime import build_live_cartridges
        from tower.routes import scene as scene_routes

        with _BlockModules("torch", "torchvision"):
            live = build_live_cartridges(settings_scene_on)

        app = FastAPI()
        app.include_router(scene_routes.router)
        app.state.live_cartridges = live
        app.state.scene_unavailable_reason = live.scene_unavailable_reason

        response = TestClient(app).get("/scene")

        assert response.status_code == 404
        assert response.json()["detail"] == live.scene_unavailable_reason
