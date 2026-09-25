"""THE GOLDEN (P4-IV RULE.md section 9.3, item 1): with `TOWER_WORLD_ANCHOR_VERIFY` unset, empty or `off`, the gate's
and the publish step's outputs -- `solution.json`, `components.json`, `consensus.json`, the gate record and
`apply_gate`'s own result, with and without the consensus's hooks -- are what the product wrote BEFORE the anchor
verification existed, recorded from the d649f9f working tree by RUN/experiments/P4-PROD/golden_record.py
(`wb_anchor_verify_fixtures.golden_outputs`). The only exclusions are the comparison's ALWAYS set (timestamps,
timings, paths)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests import wb_anchor_verify_fixtures as F
from tower.world_builder import coherence_gate as CG

GOLDEN = Path(__file__).parent / "golden" / "world_builder_gate_d649f9f.json"


def _round_floats(obj, nd=9):
    if isinstance(obj, float):
        return round(obj, nd)
    if isinstance(obj, dict):
        return {k: _round_floats(v, nd) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v, nd) for v in obj]
    return obj


@pytest.mark.parametrize("value", [None, "", "off", "  OFF "])
def test_golden_with_the_switch_unset_or_off_every_output_is_todays(tmp_path, monkeypatch, value):
    if value is None:
        monkeypatch.delenv("TOWER_WORLD_ANCHOR_VERIFY", raising=False)
    else:
        monkeypatch.setenv("TOWER_WORLD_ANCHOR_VERIFY", value)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    now = json.loads(json.dumps(F.golden_outputs(tmp_path), sort_keys=True))
    expected = golden["outputs"]
    assert sorted(now) == sorted(expected)
    for key in expected:
        if (golden["cv2"], golden["numpy"]) == (cv2.__version__, np.__version__):
            assert now[key] == expected[key], key            # value for value
        else:  # another OpenCV/NumPy build: last digits may move, nothing else may
            assert _round_floats(now[key]) == _round_floats(expected[key]), key


def test_the_gate_params_digest_is_unchanged():
    assert CG.GateParams().digest() == "6ce602286efb999f"
    assert "anchor" not in json.dumps(CG.GateParams().to_json())


def test_the_golden_exercises_what_it_claims():
    outputs = json.loads(GOLDEN.read_text(encoding="utf-8"))["outputs"]
    c = outputs["consensus_withhold"]
    assert c["record"]["consensus"]["state"] == "applied" and c["record"]["consensus"]["detached"]
    assert [e["reasons"] for e in c["components.json"]["components"]] == [[], ["seed-unstable"]]
    assert c["consensus.json"] is not None
    # the anchor's mid stretch at x2 is NOT split today (RULE.md 3.1): the case part (a) exists for
    assert [x["label"] for x in outputs["apply_gate_mid_stretch"]["components"]] == [0]
    assert "anchor_verify" not in json.dumps(outputs)
