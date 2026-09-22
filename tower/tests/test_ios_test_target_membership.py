"""Every iOS unit test on disk must be in the Xcode target that runs it.

A test nobody runs is worse than a test nobody wrote: it reports success.

`WorldPresentationTests.swift` — 52 tests, including the one asserting that
the 2026-09-09 field walk's world never says "nothing was mapped", which is
the assertion the whole World Builder remediation exists to make — was
written, committed, and **not added to the project file**. `xcodebuild test`
would have gone green having executed none of it. No error, no warning, and
opening the project in Xcode would not have added it either.

The trap is that the three targets are not alike. `Glasses` and
`GlassesUITests` are `PBXFileSystemSynchronizedRootGroup`s, where a new file
on disk is picked up automatically — so every previous iOS change taught the
lesson that adding a file is enough. `GlassesTests` is an ordinary
`PBXGroup` with an explicit ten-file list, where it is not.

This is a Python test reading an Xcode file because that is where the check
can actually run: this repository's Swift never compiles on the Windows host
that most of its work happens on, so a Swift-side guard would not fire until
the moment it is too late to be useful. `test_scene_dependency_truthfulness`
sets the precedent for reaching across.
"""

import plistlib
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROJECT = REPO / "ios" / "Glasses.xcodeproj" / "project.pbxproj"

# The targets whose membership is EXPLICIT and therefore forgettable. A
# synchronized root group needs no listing, so it is not checked here --
# and if one of these is ever converted to one, this test should be deleted
# rather than worked around.
EXPLICIT_TARGETS = {"GlassesTests": REPO / "ios" / "GlassesTests"}


def _project_text() -> str:
    if not PROJECT.exists():
        pytest.skip(f"no Xcode project at {PROJECT}")
    return PROJECT.read_text(encoding="utf-8")


@pytest.mark.parametrize("group, folder", sorted(EXPLICIT_TARGETS.items()))
def test_every_swift_file_on_disk_is_in_the_target(group, folder):
    if not folder.is_dir():
        pytest.skip(f"no {group} directory")
    text = _project_text()
    listed = set(re.findall(r"path = ([A-Za-z0-9_+-]+\.swift);", text))
    on_disk = {p.name for p in folder.glob("*.swift")}
    missing = sorted(on_disk - listed)
    assert not missing, (
        f"{group} has an explicit file list and these are not in it, so "
        f"xcodebuild would run none of them and still report success: {missing}"
    )


@pytest.mark.parametrize("group, folder", sorted(EXPLICIT_TARGETS.items()))
def test_every_listed_file_still_exists(group, folder):
    """The other direction: a listing pointing at a deleted file fails the
    build outright, which is loud — but it is the same one-line check."""
    if not folder.is_dir():
        pytest.skip(f"no {group} directory")
    text = _project_text()
    section = re.search(
        r"/\* " + re.escape(group) + r" \*/ = \{\s*isa = PBXGroup;\s*children = \((.*?)\);",
        text, re.S,
    )
    if section is None:
        pytest.skip(f"{group} is not a plain PBXGroup; membership is automatic")
    names = re.findall(r"/\* ([A-Za-z0-9_+-]+\.swift) \*/", section.group(1))
    absent = sorted(n for n in names if not (folder / n).exists())
    assert not absent, f"{group} lists files that are not on disk: {absent}"


def test_a_file_listed_in_the_group_is_also_compiled():
    """Being in the group is not being in the build.

    A file can appear in the navigator and still not be compiled: the group
    is what Xcode SHOWS, the Sources phase is what it BUILDS. Both were
    needed for the file that went missing.
    """
    text = _project_text()
    group = re.search(
        r"/\* GlassesTests \*/ = \{\s*isa = PBXGroup;\s*children = \((.*?)\);", text, re.S
    )
    if group is None:
        pytest.skip("GlassesTests is not a plain PBXGroup")
    shown = set(re.findall(r"/\* ([A-Za-z0-9_+-]+\.swift) \*/", group.group(1)))
    compiled = set(re.findall(r"/\* ([A-Za-z0-9_+-]+\.swift) in Sources \*/,", text))
    not_compiled = sorted(shown - compiled)
    assert not not_compiled, (
        "these are in the GlassesTests group but in no Sources phase, so they "
        f"are shown in Xcode and never run: {not_compiled}"
    )


def test_the_project_file_is_still_parseable():
    """A hand-edited pbxproj that no longer parses is a broken checkout.

    These four entries were added by a script on a host with no Xcode, so
    the cheapest possible structural check earns its place.
    """
    text = _project_text()
    assert text.count("{") == text.count("}"), "unbalanced braces in project.pbxproj"
    assert text.count("(") == text.count(")"), "unbalanced parens in project.pbxproj"
    # Every id in a Sources phase must name a PBXBuildFile.
    for phase in re.findall(r"isa = PBXSourcesBuildPhase;.*?files = \((.*?)\);", text, re.S):
        for identifier in re.findall(r"\b([0-9A-F]{24}) /\*", phase):
            assert re.search(
                rf"\b{identifier} /\* .*? \*/ = \{{isa = PBXBuildFile;", text
            ), f"{identifier} is compiled but has no PBXBuildFile"
