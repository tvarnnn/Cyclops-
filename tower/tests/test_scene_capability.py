"""Scene Understanding is offered when this Tower can actually run it.

Nobody using the product should have to know an environment variable
exists. `TOWER_SCENE_UNDERSTANDING` is now a tri-state:

    unset / "auto"   offered when the optional [ml] extra is installed,
                     which is decided by CONSTRUCTING the session -- the
                     same eager `import torch, torchvision` that
                     2026-08-27 measured as the only honest probe
    "on" / "true"    the same construction; a failure names itself as a
                     failure of something the operator switched on
    "off" / "false"  never constructed, and the reason says it is off

`available: true` still never promises Start succeeds -- bad weights and
a vanished CUDA still land as `state: "failed"` -- and every reason still
passes through `client_safe_reason`, so nothing here can put a filesystem
path on an unauthenticated wire.
"""

import pytest

from tests.test_scene_dependency_truthfulness import _BlockModules
from tower.results import registry


def _settings(monkeypatch, value):
    from tower.config import get_settings

    if value is None:
        monkeypatch.delenv("TOWER_SCENE_UNDERSTANDING", raising=False)
    else:
        monkeypatch.setenv("TOWER_SCENE_UNDERSTANDING", value)
    monkeypatch.delenv("TOWER_SCENE_DEVICE", raising=False)
    return get_settings()


def _offer(live):
    declaration = registry.declare(
        None,
        scene_enabled=live.scene is not None,
        scene_unavailable_reason=live.scene_unavailable_reason,
    )
    return next(
        entry
        for entry in declaration["cartridges"]
        if entry["cartridge"] == "scene_understanding"
    )


class TestTheModeIsReadHonestly:
    @pytest.mark.parametrize("value", [None, "", "  ", "auto", "AUTO"])
    def test_unset_and_auto_mean_auto(self, monkeypatch, value):
        assert _settings(monkeypatch, value).scene_understanding_mode == "auto"

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "On"])
    def test_every_spelling_of_true_means_on(self, monkeypatch, value):
        assert _settings(monkeypatch, value).scene_understanding_mode == "on"

    @pytest.mark.parametrize("value", ["0", "false", "no", "off"])
    def test_every_spelling_of_false_means_off(self, monkeypatch, value):
        assert _settings(monkeypatch, value).scene_understanding_mode == "off"

    def test_garbage_means_off_not_on(self, monkeypatch):
        """A typo must not switch a people detector on. It switches it off
        and the log says so."""
        assert _settings(monkeypatch, "maybe").scene_understanding_mode == "off"

    def test_the_boolean_view_follows_the_mode(self, monkeypatch):
        assert _settings(monkeypatch, "on").scene_understanding is True
        assert _settings(monkeypatch, "off").scene_understanding is False

    def test_the_device_defaults_to_auto(self, monkeypatch):
        """CUDA when it is there, CPU when it is not. A person should not
        have to know which GPU the Tower has to get the detector that fits."""
        assert _settings(monkeypatch, None).scene_device == "auto"


class TestAutoOffersItWhenItCanRun:
    def test_a_host_with_the_ml_extra_offers_it_by_default(self, monkeypatch):
        from tower.cartridge_runtime import build_live_cartridges

        live = build_live_cartridges(_settings(monkeypatch, None))
        assert live.scene is not None
        assert _offer(live)["available"] is True
        assert _offer(live)["unavailable_reason"] is None

    def test_a_host_without_the_ml_extra_says_what_is_missing(self, monkeypatch):
        from tower.cartridge_runtime import build_live_cartridges

        settings = _settings(monkeypatch, None)
        with _BlockModules("torch", "torchvision"):
            live = build_live_cartridges(settings)

        assert live.scene is None
        offer = _offer(live)
        assert offer["available"] is False
        reason = offer["unavailable_reason"]
        # Names what is missing, not a variable nobody set.
        assert "torch" in reason
        assert "[ml]" in reason
        assert "unset or off" not in reason
        assert "switched off" not in reason
        # Still offered: this build implements the contract.
        assert offer["contract"] == "scene_understanding.live/2026-08-27"
        assert "This build implements the contract" in reason

    def test_auto_does_not_pay_for_torch_when_it_is_off(self, monkeypatch):
        """`off` must not import torch at all: a Tower an operator switched
        off should boot as fast as one with no [ml] extra."""
        import sys

        from tower.cartridge_runtime import build_live_cartridges

        settings = _settings(monkeypatch, "off")
        with _BlockModules("torch", "torchvision"):
            live = build_live_cartridges(settings)
            assert "torch" not in sys.modules
        assert live.scene is None


class TestOffMeansOff:
    def test_off_is_never_constructed_and_the_reason_says_off(self, monkeypatch):
        from tower.cartridge_runtime import build_live_cartridges

        live = build_live_cartridges(_settings(monkeypatch, "off"))
        assert live.scene is None
        offer = _offer(live)
        assert offer["available"] is False
        assert offer["unavailable_reason"] == registry.SCENE_DISABLED_REASON
        assert "TOWER_SCENE_UNDERSTANDING" in offer["unavailable_reason"]
        assert "off" in offer["unavailable_reason"]
        # "unset" no longer means off, and the sentence must not say so.
        assert "unset" not in offer["unavailable_reason"]


class TestOnStillNamesAFailureAsAFailure:
    def test_on_with_no_torch_blames_the_dependency_not_the_variable(
        self, monkeypatch
    ):
        from tower.cartridge_runtime import build_live_cartridges

        settings = _settings(monkeypatch, "on")
        with _BlockModules("torch", "torchvision"):
            live = build_live_cartridges(settings)

        reason = _offer(live)["unavailable_reason"]
        assert "enabled" in reason
        assert "torch" in reason
        assert "unset or off" not in reason


class TestEverySurfaceAgreesInAuto:
    def test_the_route_and_the_declaration_give_one_reason(self, monkeypatch):
        """`/scene` 404 detail and `/cartridges` unavailable_reason must be
        the same sentence in auto, exactly as they already were for on."""
        from fastapi.testclient import TestClient

        from tower import cartridge_runtime
        from tower.main import create_app

        monkeypatch.delenv("TOWER_SCENE_UNDERSTANDING", raising=False)
        monkeypatch.delenv("TOWER_WORLD_ROOT", raising=False)

        def _broken(settings):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")

        monkeypatch.setattr(cartridge_runtime, "_scene_session", _broken)
        with TestClient(create_app()) as client:
            declared = next(
                entry
                for entry in client.get("/cartridges").json()["cartridges"]
                if entry["cartridge"] == "scene_understanding"
            )
            route = client.get("/scene")

        assert declared["available"] is False
        assert route.status_code == 404
        assert route.json()["detail"] == declared["unavailable_reason"]
        assert "torch" in declared["unavailable_reason"]
