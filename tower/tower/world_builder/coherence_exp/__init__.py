"""Coherence experiments: a solver-variant driver over frozen worlds.

EXPERIMENT CODE. Nothing on the builder path imports this package; it exists
so that every solver variant of the 2026-09-23 coherence run (masked
features, solver-only frames, bridge matches, the rigid truthfulness gate) is
produced by ONE driver whose off/off configuration IS the product recipe
(`global_solve.solve`), and is scored by the one harness
(`coherence_eval`).

  driver.py  VariantConfig, staging in capture order, SIFT + sequential(+loop)
             matching, the augment hook, seeded mapping (one process and one
             database copy per seed), export in the harness interchange format
  masks.py   transient (hand / arm / held phone) masks on the staged images,
             cached image-only and reused by every arm
  gate.py    the rigid gate: rigid groups by shared 3-D points, seed-ensemble
             placement spread, split what is not both coupled and stable
"""
