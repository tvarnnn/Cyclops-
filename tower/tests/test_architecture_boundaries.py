"""Import-direction invariants between the platform and its cartridges.

These are cheap to state and expensive to discover by hand. A cartridge
that leaks into shared transport does not fail loudly -- it just quietly
makes the next cartridge harder to write, which nobody notices until the
next cartridge exists.
"""

import ast
import os
import pathlib

TOWER = pathlib.Path("tower")


def _env_without_tower_settings() -> dict:
    """The ambient environment, minus every `TOWER_` setting.

    The two subprocess probes below assert that `import tower.main` pulls
    in neither torch nor easyocr. `tower/main.py` ends with a module-level
    `app = create_app()`, so importing it BUILDS THE APP -- and what the
    app constructs depends on configuration. Inheriting the operator's
    shell therefore made a structural invariant depend on an environment
    variable, and the probe measured the machine it happened to run on
    rather than the code.

    That was already broken before anything in this lane touched it: with
    `TOWER_SCENE_UNDERSTANDING` on and `TOWER_SCENE_DEVICE=cuda`,
    `_resolve_device` imports torch at construction and both probes went
    red. Making Scene import torch on the default `cpu` device widened
    that from `{scene on AND device != cpu}` to `{scene on}`, which is how
    it was noticed -- but a clean environment is what these tests always
    needed, because the invariant they defend is about where imports SIT,
    not about which flags are set.

    Everything else is preserved, so the child still finds its
    interpreter, its PATH and its `sys.path`.
    """
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("TOWER_")
    }


def _imports(path: pathlib.Path) -> list[str]:
    """Every module path an import statement reaches, INCLUDING the names.

    Recording only `node.module` for an `ImportFrom` left a hole wide
    enough to drive the whole boundary through: `from tower import
    world_builder` reports the module as `tower`, so a shared module could
    import a cartridge outright and every rule below would see nothing.
    Emitting `f"{module}.{name}"` alongside closes it, because the
    predicates match on the qualified package path.

    This was parked once on the belief that fixing it would surface
    unrelated latent violations. It surfaces none: across `tower/` the
    extended form yields zero offenders.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
    return names


def _modules_outside(package: str) -> list[pathlib.Path]:
    return [
        path
        for path in TOWER.rglob("*.py")
        if package not in path.parts and "__pycache__" not in path.parts
    ]


def test_shared_code_does_not_import_a_cartridge():
    """Transport, config and the module system must not know about a cartridge.

    World Builder is one consumer of the camera, not the definition of how
    the camera behaves. The moment shared code imports it, the next
    cartridge inherits its assumptions -- keyframes over freshness,
    latency as free, motion as the signal -- every one of which is wrong
    for Accessibility, Visual Q&A or Text/Document.
    """
    offenders = []
    for path in _modules_outside("world_builder"):
        if path in _RESULT_CHANNEL_ADAPTERS:
            continue
        for name in _imports(path):
            # A bare `"world_builder" in name` also matches
            # tower.results.world_builder_geometry -- the ADAPTER, not the
            # cartridge, and a file this rule must let a non-adapter import.
            # That false positive once pushed a fix into restyling the
            # import rather than the boundary; match the qualified package
            # path instead, mirroring the predicate
            # test_the_result_channel_core_is_cartridge_blind already uses
            # below.
            if "tower.world_builder" in name:
                offenders.append(f"{path} -> {name}")

    assert offenders == []


# The result channel reports each cartridge to iOS, so SOMETHING has to
# import a cartridge or there is nothing to report. Two files may, and the
# rule below is stricter than a blanket exemption would be:
#
#   tower/results/world_builder.py   the adapter FOR that cartridge, named
#                                    after it, and the only file that may
#                                    know its record shapes
#   tower/results/__init__.py        the single wiring point, which maps a
#                                    cartridge name to its adapter
#
# Everything else in the channel -- the envelope, the publisher, the
# registry, the routes -- must stay cartridge-blind, which is what
# test_the_result_channel_core_is_cartridge_blind pins. That is the
# invariant the original rule was really protecting: the generic parts
# must not inherit one cartridge's assumptions. An adapter named after its
# cartridge cannot leak assumptions into the next one, because the next
# one gets its own file.
#
#   tower/results/world_builder_geometry.py
#                                    the geometry adapter for that same
#                                    cartridge. Separate from the status
#                                    adapter because it answers a different
#                                    question over a different transport --
#                                    HTTP, because the status socket shares
#                                    its send lock with the frame path.
_RESULT_CHANNEL_ADAPTERS = frozenset(
    {
        TOWER / "results" / "world_builder.py",
        TOWER / "results" / "world_builder_geometry.py",
        # Added 2026-09-06: the saved-worlds listing (GET /worlds), an
        # adapter over the World Builder store exactly like the geometry
        # producer above.
        TOWER / "results" / "world_builder_library.py",
        # Added 2026-09-06: the interactive viewer (GET /worlds/{id}/render),
        # the same shape again -- an adapter over the store, named after
        # the cartridge, serving HTML where the others serve JSON.
        TOWER / "results" / "world_builder_render.py",
        TOWER / "results" / "__init__.py",
        # Added 2026-08-27 with the Scene Understanding and Document
        # Memory wire paths. Same shape, same rule: one adapter per
        # cartridge, named after it, and it is the only file outside the
        # cartridge's own package that may know its record shapes.
        TOWER / "results" / "scene_understanding.py",
        TOWER / "results" / "document_memory.py",
    }
)

# Files that may know a LIVE cartridge exists, as opposed to a persisted
# one. A separate set from the adapters above because they are exempted
# for a different reason and the difference matters.
#
# An adapter reads a record shape. These two RUN something: they construct
# a session object that owns a worker thread and a model. That is a
# stronger permission and it is granted to exactly two files.
#
#   tower/cartridge_runtime.py  the single factory `main.py` calls
#                               generically -- `main.py` asks for "the
#                               live cartridges this configuration
#                               enables" and is told, exactly as it
#                               already asks `tower.results` for a hub.
#                               Keeping the cartridge names out of
#                               `main.py` is not a dodge around
#                               test_scene_understanding_is_not_registered
#                               _as_a_production_module; it is what that
#                               test is protecting -- `main.py:68` is
#                               explicitly "the ONE place in the web
#                               process that knows a world builder
#                               exists", and the answer to a second
#                               cartridge is to stop adding places, not
#                               to add one.
#
#   tower/routes/scene.py       the control surface. Named after the
#   tower/routes/documents.py   QUESTION, not the cartridge, exactly as
#                               `geometry.py` and `observations.py` are.
_LIVE_CARTRIDGE_WIRING = frozenset(
    {
        TOWER / "cartridge_runtime.py",
        TOWER / "routes" / "scene.py",
        TOWER / "routes" / "documents.py",
    }
)

_CARTRIDGE_AWARE_FILES = _RESULT_CHANNEL_ADAPTERS | _LIVE_CARTRIDGE_WIRING


def test_the_result_channel_core_is_cartridge_blind():
    """The generic half of the result channel must import no cartridge.

    Stronger than the rule it replaces. The envelope, publisher, registry
    and routes are what the next three cartridges will publish through, so
    a single import of `world_builder` there would bake one cartridge's
    shape into the shared surface -- and this time the surface is a WIRE
    CONTRACT, where that mistake is not refactorable once a phone ships
    against it.
    """
    cartridges = ("world_builder", "object_memory", "document_memory", "scene")
    core = [
        TOWER / "results" / "envelope.py",
        TOWER / "results" / "publisher.py",
        TOWER / "results" / "registry.py",
        TOWER / "results" / "contracts.py",
        TOWER / "routes" / "results_ws.py",
        TOWER / "routes" / "cartridges.py",
        # The session control surface. Added to the CORE list rather
        # than exempted: it is the first mutating route in this Tower and
        # it addresses a cartridge by an id in the URL path, which is
        # exactly the shape that invites an import "just to look one up".
        TOWER / "routes" / "sessions.py",
    ]
    offenders = []
    for path in core:
        for name in _imports(path):
            for cartridge in cartridges:
                if f"tower.{cartridge}" in name:
                    offenders.append(f"{path.name} -> {name}")

    assert offenders == []


def test_the_result_channel_never_writes():
    """A reporting surface must not write, and must not build.

    The web process does not build worlds; a separate process does. If
    this package ever acquired a write, the Tower would have two writers
    to one store -- and on Windows the store's own docstring records that
    the second one fails with a PermissionError rather than corrupting
    quietly, which is a better failure and still a broken system.
    """
    forbidden = (
        "write_json_atomic",
        "append_jsonl",
        "append_event",
        "append_keyframe",
        "write_world",
        "write_session",
        "write_derived",
        "purge_world",
        "clear_derived",
        "acquire_writer_lock",
        "start_session",
        "stop_session",
        "observe",
        "build",
        "write_text",
        "write_bytes",
        "mkdir",
    )
    offenders = []
    for path in (TOWER / "results").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "attr", None) or getattr(func, "id", None)
                if name in forbidden:
                    offenders.append(f"{path.name} calls {name}")

    assert offenders == []


# Every cartridge package under tower/. The rule below is applied to
# every ORDERED PAIR of these, which is the point: the version that
# checked only world_builder -> object_memory left five of the six pairs
# unguarded, including both directions between the two cartridges most
# likely to want each other (object_memory and world_builder, for "where
# did I leave my keys").
_CARTRIDGE_PACKAGES = ("world_builder", "object_memory", "document_memory", "scene")


def test_a_cartridge_does_not_import_another_cartridge():
    """Rule 6: modules own their data.

    A shared VOCABULARY may be promoted to tower/ (Confidence was), but one
    cartridge reaching into another's namespace couples their schemas: a
    change on one side silently invalidates persisted records on the other.

    Symmetric and exhaustive, deliberately. The asymmetric version could
    not have caught the import a spatial-context feature actually wants,
    which is `object_memory` reaching into `world_builder` -- the
    direction it did not check. When two cartridges genuinely need to
    meet, they meet outside `tower/`: a script joining them by a frame
    identity that shared transport owns is not a coupling, because
    neither package learns the other's record shapes.
    """
    offenders = []
    for package in _CARTRIDGE_PACKAGES:
        directory = TOWER / package
        if not directory.exists():
            continue
        others = [other for other in _CARTRIDGE_PACKAGES if other != package]
        for path in directory.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            for name in _imports(path):
                for other in others:
                    if f"tower.{other}" in name or name.startswith(f"{other}."):
                        offenders.append(f"{path} -> {name}")

    assert offenders == []


def test_shared_storage_primitives_have_no_cartridge_dependency():
    """tower/storage.py and tower/capture.py are platform infrastructure.

    capture.py in particular is armed by shared transport, so versioning
    its records with a cartridge's schema constant would let a
    geometry-driven bump invalidate capture journals that have nothing to
    do with geometry.
    """
    offenders = []
    for name in ("storage.py", "capture.py", "confidence.py"):
        for imported in _imports(TOWER / name):
            if "world_builder" in imported or "object_memory" in imported:
                offenders.append(f"{name} -> {imported}")

    assert offenders == []


def _code_without_comments_or_strings(path: pathlib.Path) -> str:
    """Source with every comment and string literal removed.

    Lets a boundary be asserted against what the module DOES rather than
    what it says. The rule below used to be a substring scan over the raw
    file, which a comment naming a cartridge could trip -- a false
    positive that pushes people towards writing evasive prose instead of
    thinking about the boundary.
    """
    import io
    import tokenize

    source = path.read_text(encoding="utf-8")
    kept = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def test_world_builder_is_not_registered_as_a_production_module():
    """The integration boundary, asserted rather than assumed.

    Registration is blocked behind V1.0 (registry generalisation) and V1.1
    (lifecycle hardening), both of which are untriggered or blocked. If
    this test ever fails, someone has crossed that boundary -- which may
    be correct, but must be a deliberate decision rather than a drift.

    **What changed, and why this rule is now narrower in one place and
    stricter in another.** `main.py` supervises a per-capture worker
    PROCESS (see `capture_workers.py`), and the argv it builds names
    `scripts/world_build_session.py`. That is a deliberate crossing: the
    web process now knows that something can be run against a capture.

    It is not registration, and the distinction is the one the module
    system cares about. A registered module is loaded into this process,
    joins the frame path, and shares its lifecycle and its failure
    domain. A supervised child shares none of those -- which is exactly
    why an expensive rebuild can run repeatedly mid-session without the
    frame path noticing, the property `docs/agent-handoffs/WORLD-BUILDER.md`
    section 1 exists to protect.

    So the rule is now: main.py may name a worker as a COMMAND, and may
    say so in a comment, but must not import the cartridge, must not
    register it as a module, and must not reach a cartridge's names in
    executable code. Checked against code with comments and string
    literals stripped, which is stricter than the old raw substring scan
    for everything that actually runs.
    """
    main = TOWER / "main.py"

    for imported in _imports(main):
        assert "world_builder" not in imported, (
            f"main.py imports {imported}: the web process must not import a "
            "cartridge. Run it as a subprocess or report it through an "
            "adapter."
        )

    code = _code_without_comments_or_strings(main)
    assert "world_builder" not in code, (
        "main.py reaches a World Builder name in executable code. Naming a "
        "script path in an argv string is allowed; touching the package is "
        "not."
    )
    assert not (TOWER / "modules" / "world_builder.py").exists()


def test_the_capture_worker_supervisor_is_cartridge_blind():
    """The generic half must stay generic, or the next worker inherits this one.

    `capture_workers.py` is shared machinery: the second thing that ever
    wants to watch a capture -- an offline re-encoder, a Document Memory
    pass over selected stills -- gets it for free only if it contains no
    trace of the first. It runs an argv. It must not know what the argv
    computes.

    Stricter than the module-wide rule above, and deliberately so: this
    file is new, so there is no legacy to grandfather.
    """
    path = TOWER / "capture_workers.py"
    cartridges = ("world_builder", "world_build", "object_memory",
                  "document_memory", "scene")

    for imported in _imports(path):
        for cartridge in cartridges:
            assert cartridge not in imported, f"capture_workers.py -> {imported}"

    code = _code_without_comments_or_strings(path)
    for cartridge in cartridges:
        assert cartridge not in code, (
            f"capture_workers.py names {cartridge!r} in executable code"
        )


def test_the_experimental_cv_lab_does_not_import_a_cartridge():
    """The Lab must not depend on World Builder or Object Memory.

    Its job is to MEASURE things like blur rejection, feature yield and
    parallax -- the same questions World Builder answers privately. If it
    imported that answer it would be restating a cartridge's opinion
    rather than measuring the underlying property, and a change on the
    cartridge side would silently move a measurement.

    It would also invert the dependency the platform is built on: a
    sandbox may be thrown away, and nothing that can be thrown away should
    be upstream of a persistent world.
    """
    offenders = []
    paths = list((TOWER / "experiments").rglob("*.py"))
    paths.append(TOWER / "modules" / "experimental_cv.py")
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        for name in _imports(path):
            if "world_builder" in name or "object_memory" in name:
                offenders.append(f"{path} -> {name}")

    assert offenders == []


def test_object_memory_does_not_import_the_experimental_cv_lab():
    """The producer owns its detector; it does not borrow the Lab's.

    Same reason `tower/scene/detect.py` records and the scene rule below
    already enforces: the Lab measured these exact weights, but its
    `ExperimentResult` is a scalar plus a name->number bag and cannot
    carry a box, and a memory needs the individual detection rather than
    a count. The stronger half is the direction -- the Lab is a sandbox
    that may be thrown away, and nothing that can be thrown away should
    be upstream of a PERSISTENT store.

    A third consumer appeared, and the fix this docstring named is the
    one that was taken: the detector seam is now `tower/detection.py`,
    promoted to the platform exactly as `Confidence` was, and this
    cartridge imports it from there. That changes nothing about the rule
    below. Depending on a platform module and depending on a sandbox are
    different acts -- shared code is maintained and its boundary is
    tested (`test_the_shared_detector_imports_no_cartridge`), a sandbox
    may be deleted tomorrow -- so an import of `tower.experiments` from
    here is still forbidden, and still for the direction rather than the
    duplication.
    """
    offenders = []
    for path in (TOWER / "object_memory").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for name in _imports(path):
            if name.startswith("tower.experiments"):
                offenders.append(f"{path} -> {name}")

    assert offenders == []


def test_the_shared_detector_imports_no_cartridge():
    """`tower/detection.py` is platform code, so the arrows point one way.

    The detector seam was duplicated in two cartridges and promoted here
    once a third consumer of the same weights appeared. A promotion is
    only safe while the promoted module stays ignorant of who calls it:
    the moment this file imported Object Memory to reach a record shape,
    or Scene Understanding to reach its `BoundingBox`, every other
    cartridge would inherit that one's assumptions -- and the two
    cartridges would be coupled THROUGH the platform, which is precisely
    the coupling `test_a_cartridge_does_not_import_another_cartridge`
    forbids directly. Promoting shared code must not open a side door
    into a rule that is otherwise airtight.

    The Lab is on the list for the additional reason the cartridge rules
    already give: it is a sandbox that may be thrown away, and nothing
    that may be thrown away belongs upstream of anything.
    """
    offenders = []
    for name in _imports(TOWER / "detection.py"):
        for cartridge in _CARTRIDGE_PACKAGES:
            if f"tower.{cartridge}" in name or name.startswith(f"{cartridge}."):
                offenders.append(f"detection.py -> {name}")
        if name.startswith("tower.experiments"):
            offenders.append(f"detection.py -> {name}")

    assert offenders == []


def test_the_shared_detector_holds_no_model_and_no_registry():
    """Code was promoted; model residency deliberately was not.

    Each cartridge still loads its own 13.4 MB of weights. That is the
    property that stops a shared module becoming a single point of
    failure: a cache would give one cartridge's crash, or one
    cartridge's `release()`, a way to reach another's detector, and an
    eviction policy would give it a way to reach one MID-FRAME.

    A model manager may well be worth building later -- when a
    measurement shows contention, which nothing in this repo does today.
    This test is what makes that a decision rather than a drift: it fails
    the moment module-level mutable state appears here.
    """
    tree = ast.parse((TOWER / "detection.py").read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp)):
            offenders.append(ast.dump(node)[:60])
        if isinstance(value, ast.Call):
            called = getattr(value.func, "id", None) or getattr(
                value.func, "attr", None
            )
            if called in ("dict", "list", "set", "defaultdict", "lru_cache", "cache"):
                offenders.append(f"module-level {called}()")

    assert offenders == []


def test_no_experiment_persists_anything():
    """The Lab's descriptor declares persists_data=False. Keep it true.

    The descriptor is what `06-PRIVACY-DATA.md` is enforced against, so an
    experiment that quietly wrote to disk would not merely be untidy -- it
    would make the module's declared data behaviour a lie.

    Dataset recording is not an exception to this: it belongs to the
    SHARED recorder (`tower/capture.py`), armed by the transport, not to
    an experiment.
    """
    forbidden = ("write_json_atomic", "append_jsonl", "WorldStore", "ObservationStore")
    offenders = []
    for path in (TOWER / "experiments").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                target = node.func
                called = getattr(target, "id", None) or getattr(target, "attr", None)
                if called in forbidden or called in ("open",):
                    offenders.append(f"{path.name} calls {called}()")

    assert offenders == []


def test_shared_code_does_not_import_an_experiment_implementation():
    """`tower/routes` and `tower/frames.py` must stay experiment-agnostic.

    The transport may know the SHAPE of a result; it must not know which
    experiment produced it. A transport that special-cased `depth` would
    make the next experiment a transport change.
    """
    offenders = []
    for name in ("routes/ws.py", "routes/health.py", "frames.py", "metrics.py"):
        for imported in _imports(TOWER / name):
            if imported.startswith("tower.experiments."):
                offenders.append(f"{name} -> {imported}")

    assert offenders == []


def test_importing_the_lab_does_not_import_torch():
    """The optional [ml] extra must stay optional.

    Two experiments need torch; six do not, and the Tower must start on a
    machine that has never installed it. Every torch import is therefore
    inside a function, and this is the only way to check that -- an
    in-process assertion would pass merely because some earlier test in
    the same session had already imported it.

    Scene Understanding is switched OFF for the probe, deliberately. Since
    2026-09-07 it is offered automatically on a host that has the [ml]
    extra, and offering it means constructing its session at boot, which
    imports torch on purpose (`cartridge_runtime._scene_session` explains
    why that import is eager). That is the cartridge's decision, not the
    Lab's, and this test is about the Lab. A host WITHOUT the extra is
    covered by `tests/test_scene_capability.py`, which blocks torch and
    asserts the Tower still builds and says what is missing.
    """
    import subprocess
    import sys

    probe = (
        "import sys, tower.main, tower.experiments; "
        "print([m for m in ('torch','torchvision','timm') if m in sys.modules])"
    )
    env = _env_without_tower_settings()
    env["TOWER_SCENE_UNDERSTANDING"] = "off"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("[]"), result.stdout


def test_document_memory_does_not_import_another_cartridge():
    """Document Memory must not reach into World Builder or Object Memory.

    Two reasons, and the first is the sharper one. World Builder's blur
    and motion gates would reject exactly the frames this cartridge wants:
    a held-still, high-detail view of a page has near-zero parallax and is
    `insufficient_motion` to a mapper. Document Memory INVERTS World
    Builder's signal, so sharing that code would mean sharing an
    assumption that is wrong here.

    Second: the brief forbids fabricating spatial anchors. If this module
    could see a world, someone would eventually derive one instead of
    requiring a caller to supply it.
    """
    offenders = []
    for path in (TOWER / "document_memory").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for name in _imports(path):
            if "world_builder" in name or "object_memory" in name:
                offenders.append(f"{path} -> {name}")

    assert offenders == []


def test_document_memory_is_not_registered_as_a_production_module():
    """The same integration boundary World Builder stops at.

    The module contract is a registry of one with a scalar-shaped result,
    and OCR costs ~1.2s per page -- it could not sit on the event loop
    even if a slot were free. If this test ever fails, someone crossed the
    V1.0/V1.1 boundary, which may be correct but must be deliberate.
    """
    main = (TOWER / "main.py").read_text(encoding="utf-8")

    assert "document_memory" not in main
    assert not (TOWER / "modules" / "document_memory.py").exists()


def test_the_ocr_dependency_is_not_imported_at_module_load():
    """easyocr is an optional [ocr] extra; the Tower must start without it.

    Checked in a subprocess: an in-process assertion would pass merely
    because some earlier test had already imported it.
    """
    import subprocess
    import sys

    probe = (
        "import sys, tower.main, tower.document_memory.engine, "
        "tower.document_memory.ocr, tower.document_memory.retrieval; "
        "print([m for m in ('easyocr','torch','scipy','skimage') "
        "if m in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=_env_without_tower_settings(),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("[]"), result.stdout


def test_shared_code_does_not_import_document_memory():
    """Transport and the module system must not know this cartridge exists.

    Exempts the same small set the World Builder rule exempts, for the
    same stated reason: something has to import a cartridge or there is
    nothing to report, and an adapter named after its cartridge cannot
    leak that cartridge's assumptions into the next one.

    The match narrowed from a bare `"document_memory" in name` to the
    qualified package path when the adapter arrived, and that is not a
    weakening -- it is the identical correction the World Builder rule
    already carries above: a bare substring also matches
    `tower.results.document_memory`, which is the ADAPTER, and a rule that
    cannot tell an adapter from its cartridge pushes the fix into
    restyling the import rather than into the boundary.
    """
    offenders = []
    for path in _modules_outside("document_memory"):
        if path in _CARTRIDGE_AWARE_FILES:
            continue
        for name in _imports(path):
            if "tower.document_memory" in name:
                offenders.append(f"{path} -> {name}")

    assert offenders == []


# The two files that may know Object Memory's shapes, for the same
# reason `_RESULT_CHANNEL_ADAPTERS` exists above: the result channel has
# to import SOMETHING or there is nothing to report, and an adapter named
# after its cartridge cannot leak that cartridge's assumptions into the
# next one, because the next one gets its own file.
#
#   tower/results/object_memory.py   the adapter, and the only module
#                                    outside the cartridge that may know
#                                    its record shapes or its policy
#   tower/routes/observations.py     the route, which imports only the
#                                    adapter and never the cartridge
_OBJECT_MEMORY_ADAPTERS = frozenset({TOWER / "results" / "object_memory.py"})


def test_shared_code_does_not_import_object_memory():
    """The rule that was missing while the other three cartridges had one.

    Written after `main.py` acquired `from tower.object_memory.relevance
    import recordable_classes` inside a function and no test noticed. The
    wiring point knows the world builder as an argv and must know this
    cartridge the same way; what it needs from the policy travels through
    the adapter as a tuple of strings.

    Function-level imports count. The AST walk sees them, and hiding an
    import inside a function is the commonest way a boundary is crossed
    while looking untouched at the top of the file.
    """
    offenders = []
    for path in _modules_outside("object_memory"):
        if path in _OBJECT_MEMORY_ADAPTERS:
            continue
        for name in _imports(path):
            if "tower.object_memory" in name:
                offenders.append(f"{path} -> {name}")

    assert offenders == []


def test_the_object_memory_route_reaches_only_its_adapter():
    """`tower/routes/observations.py` is a route, not a second adapter.

    Stricter than the rule above, and deliberately: this file is the one
    most likely to acquire a "just to look up a class name" import,
    because it is where the question is asked.
    """
    path = TOWER / "routes" / "observations.py"

    for name in _imports(path):
        assert "tower.object_memory" not in name, f"observations.py -> {name}"


def test_the_evidence_behind_the_class_policy_travels_with_it():
    """A tier is a claim, and a claim needs its sample size attached.

    `PERSISTED_CLASSES` was once a bare two-name tuple justified by a
    comment. Its replacement carries counts, so a later corpus can be
    compared against it and a reviewer can see what a tier was decided
    on rather than trusting that it was decided on something.
    """
    import sys

    sys.path.insert(0, ".")
    from tower.object_memory.classes import CLASS_EVIDENCE, REMEMBERED

    remembered = [
        name
        for name, evidence in CLASS_EVIDENCE.items()
        if evidence.tier == REMEMBERED
    ]
    assert remembered, "no class is remembered outright; the guard is vacuous"
    for name in remembered:
        evidence = CLASS_EVIDENCE[name]
        assert evidence.inspected > 0, (
            f"{name} is remembered on the detector's word alone with no "
            "crop ever inspected"
        )
        assert evidence.precision == 1.0, name


def test_scene_understanding_does_not_import_another_cartridge():
    """Scene Understanding must not import the Lab, World Builder or Object Memory.

    The Lab measured the exact detector this cartridge uses, which is what
    the promotion path is for -- but the Lab's `ExperimentResult` is
    scalars and a name->number bag and cannot carry a box. The two want
    different things from the same weights: the Lab wants swappable models
    with timings, this wants stable structured output. Importing would
    couple a sandbox that may be thrown away to a production consumer.
    """
    offenders = []
    for path in (TOWER / "scene").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        for name in _imports(path):
            if any(
                cartridge in name
                for cartridge in ("world_builder", "object_memory", "document_memory")
            ):
                offenders.append(f"{path} -> {name}")
            if name.startswith("tower.experiments"):
                offenders.append(f"{path} -> {name}")

    assert offenders == []


def test_scene_understanding_persists_nothing():
    """Its strongest privacy property, enforced rather than intended.

    A cartridge answering "what is around me NOW" has no reason to write
    to disk, and doing so would import all of Environmental Memory's
    retention and purge surface for no gain. There is no store here, and
    there must not become one by accident.
    """
    # An adversarial review pointed out that an earlier version of this
    # list only named the project's OWN write helpers, so
    # `Path.write_text`, `cv2.imwrite`, `np.save` or `pickle.dump` would
    # all have slipped through. Nothing here calls them today; the
    # enforcement should not depend on that continuing by luck.
    forbidden = (
        "write_json_atomic",
        "append_jsonl",
        "WorldStore",
        "ObservationStore",
        "DocumentStore",
        "open",
        "write_text",
        "write_bytes",
        "imwrite",
        "save",
        "savez",
        "savetxt",
        "dump",
        "to_csv",
        "mkdir",
        "makedirs",
        "touch",
    )
    offenders = []
    for path in _scene_wire_path():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if called in forbidden:
                    offenders.append(f"{path.name} calls {called}()")

    assert offenders == []


def _scene_wire_path() -> list:
    """Every file between a frame and this cartridge's payload.

    Widened 2026-08-27, and the widening is the point. Scanning only
    `tower/scene/**` was sufficient while the cartridge had no consumer:
    there was nowhere else for a write to hide. The moment a live session
    is published, "publish" can quietly become "buffer to disk" in the
    adapter, the runtime factory or the route -- none of which the old
    glob saw -- and that is the single most likely way this change
    destroys the cartridge's best property.

    `tower/routes/scene.py` is included even though it is a route, because
    a route is exactly where somebody would cache a payload to a file to
    make a dashboard faster.
    """
    paths = [
        path
        for path in (TOWER / "scene").rglob("*.py")
        if "__pycache__" not in path.parts
    ]
    for extra in (
        TOWER / "results" / "scene_understanding.py",
        TOWER / "routes" / "scene.py",
        TOWER / "cartridge_runtime.py",
    ):
        if extra.exists():
            paths.append(extra)
    return paths


def test_scene_understanding_is_not_registered_as_a_production_module():
    main = (TOWER / "main.py").read_text(encoding="utf-8")

    assert "tower.scene" not in main
    assert not (TOWER / "modules" / "scene.py").exists()


def test_shared_code_does_not_import_scene_understanding():
    """Same rule, same exemption set, same reason as World Builder's.

    Note what is NOT exempted and must never be: `results/envelope.py`,
    `results/publisher.py`, `results/registry.py`, `results/contracts.py`,
    `routes/results_ws.py` and `routes/cartridges.py` are pinned
    separately by `test_the_result_channel_core_is_cartridge_blind`, and
    that test names `scene` explicitly. The generic half of the channel
    stays blind whatever is exempted here.
    """
    offenders = []
    for path in _modules_outside("scene"):
        if path in _CARTRIDGE_AWARE_FILES:
            continue
        for name in _imports(path):
            if name.startswith("tower.scene"):
                offenders.append(f"{path} -> {name}")

    assert offenders == []


def test_no_cartridge_claims_gaze_or_persistent_identity():
    """Two words the platform may never use about a person.

    `07-PLATFORM-CONSTRAINTS.md` Limitation 8: the camera cannot establish
    that anyone looked at anything, so "gaze" and "looking_at" are claims
    no sensor here supports. Persistent identity is forbidden outright by
    the cartridge brief.

    A grep-shaped test, deliberately: this is about the vocabulary that
    reaches a consumer, and vocabulary is exactly what drifts.
    """
    banned = ("looking_at", "gaze_direction", "is_looking", "face_id", "person_id")
    offenders = []
    for package in ("scene", "document_memory", "object_memory", "world_builder"):
        for path in (TOWER / package).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    names.add(node.attr)
                elif isinstance(node, ast.arg):
                    names.add(node.arg)
                elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    names.add(node.name)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    # ANY string containing a banned word, not only one
                    # equal to it. A JSON key or a rendered answer reaches
                    # a consumer just as an identifier does, and a review
                    # pointed out that exact-match let a substring through.
                    for word in banned:
                        if word in node.value:
                            names.add(word)
                elif isinstance(node, ast.JoinedStr):
                    # An f-string can assemble a banned word from parts.
                    # Its literal segments are Constants handled above;
                    # this catches the whole rendered shape where it is
                    # statically visible.
                    literal = "".join(
                        piece.value
                        for piece in node.values
                        if isinstance(piece, ast.Constant)
                        and isinstance(piece.value, str)
                    )
                    for word in banned:
                        if word in literal:
                            names.add(word)
            for word in banned:
                if word in names:
                    offenders.append(f"{path} uses {word!r}")

    assert offenders == []
