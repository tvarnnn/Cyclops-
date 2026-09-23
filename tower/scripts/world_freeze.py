#!/usr/bin/env python
"""Freeze a saved world and its raw capture as read-only experiment evidence.

WHY THIS EXISTS

A coherence experiment is only an A/B if both arms read identical bytes.
The live store is not that: the finisher, a re-solve or a later walk can
rewrite a world in place. So every experiment reads a frozen copy, and
this tool makes one, proves it is byte-identical to its source, and
marks it read-only.

WHAT IT DOES

For each ``--world``:

  <data>/world_builder/worlds/<world>   ->  <dest>/worlds/<world>
  <data>/captures/<capture_id>          ->  <dest>/captures/<capture_id>

where each capture id is read from the world's own ``session.json``
files, then followed forward through ``continues_capture`` links so a
walk that reconnected freezes every capture it read. A sha256 manifest
of every source file is written next to the copy as
``<dest>/manifests/<world>.sha256.json``; the copy is then re-hashed and
must match it exactly, or the tool exits non-zero.

Nothing under ``--data`` is written. An existing frozen tree is never
overwritten: it is re-hashed against its source and reused only if every
byte matches, so re-running the tool is a verification, not a merge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from pathlib import Path

from tower.artifact_paths import artifact_root_arg

CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root: Path) -> dict[str, dict]:
    """Relative POSIX path -> {sha256, bytes, mtime} for every file under *root*."""
    entries: dict[str, dict] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        info = path.stat()
        entries[path.relative_to(root).as_posix()] = {
            "sha256": sha256_file(path),
            "bytes": info.st_size,
            "mtime": info.st_mtime,
        }
    return entries


def capture_ids(world_dir: Path, captures_root: Path) -> list[str]:
    """Capture ids a world's sessions read, including continuation captures.

    A session records only the capture it started on. A reconnect writes
    a new capture whose ``continues_capture`` names the previous one, so
    the chain is followed forward until no capture continues it.
    """
    ids: list[str] = []
    for session_json in sorted((world_dir / "sessions").glob("*/session.json")):
        capture = json.loads(session_json.read_text(encoding="utf-8")).get("capture_id")
        if capture and capture not in ids:
            ids.append(capture)
    continues: dict[str, str] = {}
    for capture_json in captures_root.glob("*/capture.json"):
        previous = json.loads(capture_json.read_text(encoding="utf-8")).get("continues_capture")
        if previous:
            continues[previous] = capture_json.parent.name
    for capture in list(ids):
        while capture in continues and continues[capture] not in ids:
            capture = continues[capture]
            ids.append(capture)
    return ids


def make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, stat.S_IREAD)


def freeze_one(data: Path, world: str, dest: Path) -> dict:
    world_src = data / "world_builder" / "worlds" / world
    if not world_src.is_dir():
        raise SystemExit(f"no such world: {world_src}")
    trees = {f"worlds/{world}": world_src}
    missing_captures = []
    for capture in capture_ids(world_src, data / "captures"):
        capture_src = data / "captures" / capture
        if capture_src.is_dir():
            trees[f"captures/{capture}"] = capture_src
        else:
            missing_captures.append(capture)

    manifest = {
        "schema": 1,
        "world_id": world,
        "frozen_at": time.time(),
        "source_data_root": str(data),
        "missing_captures": missing_captures,
        "trees": {},
    }
    for rel, src in trees.items():
        source_hashes = hash_tree(src)
        # An existing frozen tree is verified, never overwritten: it is
        # reused only if every byte still matches its source.
        already_frozen = (dest / rel).exists()
        if not already_frozen:
            shutil.copytree(src, dest / rel, copy_function=shutil.copy2)
        copy_hashes = hash_tree(dest / rel)
        mismatched = sorted(
            name
            for name in set(source_hashes) | set(copy_hashes)
            if source_hashes.get(name, {}).get("sha256") != copy_hashes.get(name, {}).get("sha256")
        )
        if mismatched:
            raise SystemExit(f"copy of {src} does not match its source: {mismatched[:10]}")
        make_read_only(dest / rel)
        manifest["trees"][rel] = {
            "source": str(src),
            "already_frozen": already_frozen,
            "files": len(source_hashes),
            "bytes": sum(entry["bytes"] for entry in source_hashes.values()),
            "entries": source_hashes,
        }

    manifests = dest / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    out = manifests / f"{world}.sha256.json"
    out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return {
        "world": world,
        "manifest": str(out),
        "trees": {rel: (tree["files"], tree["bytes"]) for rel, tree in manifest["trees"].items()},
        "missing_captures": missing_captures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path, help="the Tower data directory (holds world_builder/ and captures/)")
    parser.add_argument("--dest", required=True, type=artifact_root_arg, help="frozen evidence root")
    parser.add_argument("--world", action="append", required=True, help="world id; repeatable")
    args = parser.parse_args(argv)
    for world in args.world:
        print(json.dumps(freeze_one(args.data.resolve(), world, args.dest)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
