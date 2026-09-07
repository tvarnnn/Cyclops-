"""Document Memory is product-managed: a stock Tower serves it.

Before 2026-09-07 the cartridge was declared and UNAVAILABLE on every
Tower that had not set `TOWER_DOCUMENT_ROOT` by hand, and the phone told
the wearer so in the words of an environment variable. Object Memory
reversed the identical default on 2026-08-26 for the identical reason
("a default that hides data from its owner while still storing it
protects nobody"). This file holds Document Memory to the same rule.
"""

import pytest

from tower import config
from tower.config import DEFAULT_DOCUMENT_ROOT, TOWER_ROOT, get_settings


@pytest.fixture(autouse=True)
def _clean_document_environment(monkeypatch):
    for name in (
        "TOWER_DOCUMENT_ROOT",
        "TOWER_DOCUMENT_ENABLED",
        "TOWER_DOCUMENT_CAPTURE",
        "TOWER_DOCUMENT_AUTOSTART",
        "TOWER_DOCUMENT_DEVICE",
        "TOWER_DOCUMENT_RETENTION_DAYS",
    ):
        monkeypatch.delenv(name, raising=False)


class TestTheManagedRoot:
    def test_a_stock_tower_has_a_document_root(self):
        settings = get_settings()

        assert settings.document_enabled is True
        # `config.DEFAULT_DOCUMENT_ROOT`, read live, not the name bound at
        # import. The claim is "a stock Tower resolves to the product
        # default, whatever it is", and the autouse fixture in conftest
        # repoints that default at a temp directory so the suite never
        # reads the documents of whoever owns this checkout. Comparing
        # against the import-time value would assert the fixture away.
        # That the product default really is TOWER_ROOT/data/document_memory
        # is the separate claim tested below.
        assert settings.document_root == config.DEFAULT_DOCUMENT_ROOT

    def test_the_default_is_absolute_and_under_the_tower_data_directory(self):
        from pathlib import Path

        root = Path(DEFAULT_DOCUMENT_ROOT)

        assert root.is_absolute()
        assert root.parent == TOWER_ROOT / "data"
        assert root.name == "document_memory"

    def test_the_default_is_isolated_from_the_sibling_cartridges(self):
        assert DEFAULT_DOCUMENT_ROOT != config.DEFAULT_OBSERVATION_ROOT
        assert not DEFAULT_DOCUMENT_ROOT.startswith(
            config.DEFAULT_OBSERVATION_ROOT
        )

    def test_an_operator_override_still_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TOWER_DOCUMENT_ROOT", str(tmp_path / "elsewhere"))

        assert get_settings().document_root == str(tmp_path / "elsewhere")

    def test_switching_the_cartridge_off_removes_the_root(self, monkeypatch, tmp_path):
        """Off wins over an override: a Tower told not to run Document
        Memory does not get a root because a variable was left set."""
        monkeypatch.setenv("TOWER_DOCUMENT_ENABLED", "false")
        monkeypatch.setenv("TOWER_DOCUMENT_ROOT", str(tmp_path))

        settings = get_settings()

        assert settings.document_enabled is False
        assert settings.document_root is None


class TestCaptureIsOnButRecordsNothingUntilStarted:
    def test_capture_follows_the_enable_flag(self):
        """A capture session EXISTS by default. It is not RUNNING: the
        session starts only when a person starts it (`document_autostart`
        stays off), which is the standard 06-PRIVACY-DATA.md holds the
        dataset recorder to -- arming is not recording."""
        settings = get_settings()

        assert settings.document_capture is True
        assert settings.document_autostart is False

    def test_capture_can_be_switched_off_for_a_read_only_tower(self, monkeypatch):
        monkeypatch.setenv("TOWER_DOCUMENT_CAPTURE", "false")

        settings = get_settings()

        assert settings.document_capture is False
        # Live, for the reason given in TestTheManagedRoot above.
        assert settings.document_root == config.DEFAULT_DOCUMENT_ROOT

    def test_disabling_the_cartridge_disables_capture_too(self, monkeypatch):
        monkeypatch.setenv("TOWER_DOCUMENT_ENABLED", "0")
        monkeypatch.setenv("TOWER_DOCUMENT_CAPTURE", "true")

        assert get_settings().document_capture is False


class TestDeviceAndRetention:
    def test_the_ocr_device_defaults_to_auto(self):
        assert get_settings().document_device == "auto"

    @pytest.mark.parametrize("value", ["cpu", "cuda", "CUDA ", "auto"])
    def test_known_devices_are_accepted(self, monkeypatch, value):
        monkeypatch.setenv("TOWER_DOCUMENT_DEVICE", value)

        assert get_settings().document_device == value.strip().lower()

    def test_an_unknown_device_falls_back_rather_than_failing_at_first_start(
        self, monkeypatch
    ):
        monkeypatch.setenv("TOWER_DOCUMENT_DEVICE", "gpu0")

        assert get_settings().document_device == "auto"

    def test_retention_defaults_to_thirty_days_and_is_configurable(self, monkeypatch):
        assert get_settings().document_retention_days == 30.0

        monkeypatch.setenv("TOWER_DOCUMENT_RETENTION_DAYS", "7")
        assert get_settings().document_retention_days == 7.0

    def test_a_negative_retention_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("TOWER_DOCUMENT_RETENTION_DAYS", "-3")

        assert get_settings().document_retention_days == 30.0
