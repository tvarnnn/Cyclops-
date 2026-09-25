"""The anchor verification's masked pair cache survives a re-finish (review V15, MED-1).

`world_refinish.set_aside` moves `solve/<session>` aside and copies back only `SOLVE_COPY_BACK`. The pair sets under
`verify_pairs/` are keyed by the content they were built from (`anchor_verify.build_masked_pairs`), so a copied-back
entry can only be read for the exact images and masks it was made from; without the copy-back a re-finish with
`TOWER_WORLD_ANCHOR_VERIFY` on rebuilt the set (40-160 s) and RULE.md section 9.3's "content-cached" did not hold.
"""

from __future__ import annotations

from tests.test_world_builder_refinish import (  # noqa: F401 -- fixtures, the autouse one included
    S1,
    W1,
    _no_native_warm,
    _old_world,
    _Solve,
    stages,
)
from scripts import world_refinish as wr
from tower.world_builder import anchor_verify as AV


def test_the_pair_cache_is_on_the_copy_back_list():
    assert AV.PAIRS_DIRNAME == "verify_pairs" and AV.PAIRS_DIRNAME in wr.SOLVE_COPY_BACK


def test_a_refinish_round_trip_keeps_the_pair_cache(tmp_path, stages):  # noqa: F811
    store, kids = _old_world(tmp_path)
    solve_dir = store.world_dir(W1) / "solve" / S1
    entry = solve_dir / AV.PAIRS_DIRNAME / "0123456789abcdef"
    entry.mkdir(parents=True)
    for name, data in (("pairs.npz", b"pairs"), ("descriptors.npy", b"descriptors"), ("manifest.json", b"{}")):
        (entry / name).write_bytes(data)
    wr.refinish(store, tmp_path, W1, S1, solve_runner=_Solve(store, kids, gate_writes=False), stamp="v")
    aside = store.world_dir(W1) / "refinish" / "v" / "solve" / S1
    for name, data in (("pairs.npz", b"pairs"), ("descriptors.npy", b"descriptors"), ("manifest.json", b"{}")):
        assert (entry / name).read_bytes() == data, "copied back into the fresh solve"
        assert (aside / AV.PAIRS_DIRNAME / entry.name / name).read_bytes() == data, "and kept set aside"
