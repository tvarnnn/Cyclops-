import asyncio

from tower.experiments import baseline, edge_detection
from tower.modules.base import ModuleState
from tower.modules.container import ModuleContainer
from tower.modules.experimental_cv import ExperimentalCVModule


def _jpeg_bytes():
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), color=(200, 50, 50)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_descriptor_declares_no_persistence_or_transmission():
    module = ExperimentalCVModule("baseline")

    assert module.descriptor.id == "experimental-cv"
    behavior = module.descriptor.data_behavior
    assert behavior.persists_data is False
    assert behavior.retains_raw_imagery is False
    assert behavior.supports_purge is False
    assert behavior.transmits_externally is False


def test_process_with_baseline_matches_baseline_run_directly():
    module = ExperimentalCVModule("baseline")
    asyncio.run(module.load())
    asyncio.run(module.start())
    raw_bytes = _jpeg_bytes()

    result = module.process(raw_bytes)
    direct = baseline.run(raw_bytes)

    assert result.result_value == direct.result_value
    assert result.result_label == "mean_intensity"


def test_process_with_edge_detection_matches_edge_detection_run_directly():
    module = ExperimentalCVModule("edge_detection")
    asyncio.run(module.load())
    asyncio.run(module.start())
    raw_bytes = _jpeg_bytes()

    result = module.process(raw_bytes)
    direct = edge_detection.run(raw_bytes)

    assert result.result_value == direct.result_value
    assert result.result_label == "edge_density"


def test_full_lifecycle_reaches_unloaded():
    module = ExperimentalCVModule("baseline")
    asyncio.run(module.load())
    asyncio.run(module.start())
    asyncio.run(module.stop())
    asyncio.run(module.unload())

    assert module.state == ModuleState.UNLOADED


def test_unknown_experiment_name_leaves_the_module_active_and_the_lab_failed():
    """Loud, not terminal. Until 2026-09-06 this marked the module FAILED
    -- a state with no way back -- so a typo in TOWER_CV_EXPERIMENT killed
    every experiment until a restart. The Lab now reports `failed` with
    the reason and any `cv_lab_start` recovers it."""
    module = ExperimentalCVModule("not-a-real-experiment")
    container = ModuleContainer(module)

    asyncio.run(container.load_and_start())  # must not raise

    assert container.state == ModuleState.ACTIVE
    assert module.lab.status()["lifecycle"]["state"] == "failed"
    assert "not-a-real-experiment" in module.lab.status()["lifecycle"]["reason"]
