"""Durable storage for observed documents.

Append-only JSONL plus atomic whole-file rewrite for prune and purge --
the shape World Builder and Object Memory independently converged on, and
the third consumer is where a pattern stops being a coincidence.

Rewrites operate on **raw dicts**, not on parsed records. Round-tripping
through the current dataclass would silently drop any key the schema does
not yet interpret, which is precisely the promise the reserved fields
exist to keep.
"""

import logging
import threading
from pathlib import Path

from tower.document_memory.records import (
    SCHEMA_VERSION,
    DocumentObservation,
    document_observation_from_json_dict,
)
from tower.storage import append_jsonl, read_raw_jsonl, replace_with_retry

# A reader that meets a rewrite mid-flight on Windows gets PermissionError
# on open. Retried briefly: the rewrite is atomic and finishes in
# milliseconds, and a status producer that died on it reported the
# library unavailable for one tick, which is a false claim.
READ_RETRIES = 5
READ_RETRY_S = 0.02


def _read_raw_with_retry(path):
    import time

    for attempt in range(READ_RETRIES):
        try:
            return read_raw_jsonl(path)
        except PermissionError:
            if attempt == READ_RETRIES - 1:
                raise
            time.sleep(READ_RETRY_S * (attempt + 1))
    return [], 0  # pragma: no cover

logger = logging.getLogger(__name__)

DOCUMENTS_FILENAME = "documents.jsonl"
IMAGES_DIRNAME = "pages"
TEMP_SUFFIX = ".tmp"


class DocumentStoreError(Exception):
    """Base for every refusal this store makes."""


class UnsupportedSchemaError(DocumentStoreError):
    """A persisted record's schema version is not one this reader knows."""


class DocumentStore:
    """Everything Document Memory keeps, and the only thing that deletes it.

    Retention is a **window**, not "forever". Documents are the platform's
    clearest case of sensitive content, so an unbounded default would be
    the wrong one to pick by omission -- `06-PRIVACY-DATA.md` requires
    retention to be configurable rather than indefinite-by-default.
    """

    def __init__(
        self, directory, retention_seconds: float | None = None, clock=None
    ) -> None:
        import time

        if retention_seconds is not None and retention_seconds < 0:
            raise ValueError(
                f"retention_seconds must be non-negative or None, got "
                f"{retention_seconds!r}"
            )
        self._directory = Path(directory)
        self._retention_seconds = retention_seconds
        self._clock = clock or time.time
        self._path = self._directory / DOCUMENTS_FILENAME
        # Guards append against prune/purge: both replace the backing file,
        # which raises PermissionError on Windows if an append is in
        # flight and can silently lose the line on POSIX.
        self._lock = threading.Lock()

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def path(self) -> Path:
        return self._path

    @property
    def retention_seconds(self) -> float | None:
        return self._retention_seconds

    def images_dir(self) -> Path:
        return self._directory / IMAGES_DIRNAME

    # -- write ---------------------------------------------------------

    def append(self, document: DocumentObservation) -> None:
        with self._lock:
            append_jsonl(self._path, document.to_json_dict())

    def update(self, document: DocumentObservation) -> bool:
        """Replace the stored record with this id. False if there is none.

        The one write that is not an append, and it exists for one
        reason: a document seen AGAIN gains a sighting, and a sighting
        belongs on the record it is a sighting of. Rewritten the way
        prune and purge rewrite -- atomically, whole-file, on raw dicts
        -- so a crash mid-update leaves the previous file intact and an
        unknown key on some other record survives untouched.
        """
        import json
        import os

        with self._lock:
            raw_records, _ = _read_raw_with_retry(self._path)
            replaced = False
            kept = []
            for record in raw_records:
                if record.get("document_id") == document.document_id:
                    kept.append(document.to_json_dict())
                    replaced = True
                else:
                    kept.append(record)
            if not replaced:
                return False
            temp_path = self._path.with_name(self._path.name + TEMP_SUFFIX)
            try:
                with temp_path.open("w", encoding="utf-8") as handle:
                    for record in kept:
                        handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                # Retried, never bare: a status producer, a search or the
                # phone's re-fetch may hold the journal open at this
                # instant, and Windows refuses a rename onto an open file.
                replace_with_retry(temp_path, self._path)
            finally:
                temp_path.unlink(missing_ok=True)
            return True

    def write_page_image(self, filename: str, jpeg_bytes: bytes) -> Path:
        """Persist one corrected page image. Off the default path.

        The single choke point for every pixel this cartridge ever
        writes, mirroring `world_builder.store.write_keyframe_image` and
        `capture.write_frame`. Whatever redaction policy is eventually
        chosen becomes a change to one function.

        fsync before rename, so a record that references an image is
        never left pointing at a partial file.
        """
        import os

        directory = self.images_dir()
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        temp_path = path.with_name(path.name + TEMP_SUFFIX)
        try:
            with temp_path.open("wb") as handle:
                handle.write(jpeg_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            replace_with_retry(temp_path, path)
        finally:
            temp_path.unlink(missing_ok=True)
        return path

    # -- read ----------------------------------------------------------

    def _retention_cutoff(self) -> float | None:
        if self._retention_seconds is None:
            return None
        return self._clock() - self._retention_seconds

    @staticmethod
    def _is_within_retention(raw: dict, cutoff: float | None) -> bool:
        """Shared by reads and prune, so the two can never disagree.

        `recorded_at`, not `observed_at`: retention is about how long WE
        have held the data, which is the privacy-relevant clock. They are
        equal today and diverge the moment a real capture timestamp is
        threaded through.

        A missing or non-numeric `recorded_at` cannot be shown to be
        within retention, so it is treated as EXPIRED. Defaulting the
        other way would make a malformed record permanently readable,
        which is the wrong direction for a window that exists to remove
        things. (`bool` is excluded because it is an `int` subclass.)
        """
        if cutoff is None:
            return True
        recorded_at = raw.get("recorded_at")
        is_numeric = isinstance(recorded_at, (int, float)) and not isinstance(
            recorded_at, bool
        )
        return is_numeric and recorded_at >= cutoff

    def read_all(self, *, include_expired: bool = False) -> list[DocumentObservation]:
        """Every stored document still within retention, oldest first.

        A record whose schema version is unknown is **skipped with a
        warning**, not guessed at and not fatal: one unreadable record
        must not make the whole memory unreadable.

        FILTERING IS THE DEFAULT, and it was not always.

        Until 2026-08-27 this method ignored `retention_seconds`
        entirely: the window was consumed only by `prune_expired` and
        `purge`, both deletion paths. So a reader that constructed a
        store with an 86-second window was served a 400-day-old document
        in full -- while the response it fed asserted, in three separate
        strings, that the window had been applied.

        That is worse than a missing feature. A privacy control that
        reports success is what `06-PRIVACY-DATA.md` calls a false
        assurance, and it is why iOS ships no privacy toggle backed by a
        Tower it cannot verify.

        It is also a bug this repository had already fixed one cartridge
        over. `tower/object_memory/store.py:44` records the identical
        defect and the identical fix; `_is_within_retention` above is
        that method, and it is a static method for the same reason:
        reads and prune must use one definition of "expired" or the two
        drift.

        `include_expired` is the opt-out and must be asked for by name.
        It exists for maintenance paths -- a purge counting what it
        deletes, an operator auditing the file -- and is never the right
        answer for anything a wearer will be shown.
        """
        raw_records, corrupt = _read_raw_with_retry(self._path)
        if corrupt:
            logger.warning(
                "document store: skipped %s corrupt line(s) in %s",
                corrupt,
                self._path,
            )
        cutoff = None if include_expired else self._retention_cutoff()
        documents = []
        for record in raw_records:
            if not self._is_within_retention(record, cutoff):
                continue
            version = record.get("schema_version")
            if version != SCHEMA_VERSION:
                logger.warning(
                    "document store: skipping record with schema_version %r "
                    "(this reader knows %s)",
                    version,
                    SCHEMA_VERSION,
                )
                continue
            try:
                documents.append(document_observation_from_json_dict(record))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("document store: unparseable record skipped: %s", exc)
        return documents

    def read_one(
        self, document_id: str, *, include_expired: bool = False
    ) -> DocumentObservation | None:
        """One document, or None. Honours the retention window by default.

        The default matters: a fetch by id is a READ, and a read that
        ignored the window would be the whole retention bug again through
        one route -- an id is easy to come by, because a listing hands
        them out.
        """
        for document in self.read_all(include_expired=include_expired):
            if document.document_id == document_id:
                return document
        return None

    def read_recent(self, limit: int) -> list[DocumentObservation]:
        """The `limit` most recently RECORDED documents, newest last.

        Recorded order, not observed order: this is the dedup candidate
        set, and what a wearer is likely to look at again is what was
        written most recently.
        """
        documents = self.read_all()
        return documents[-limit:] if limit > 0 else []

    def count(self, *, include_expired: bool = False) -> int:
        # The keyword was accepted and dropped for one commit. A privacy
        # opt-out that reports the wrong answer without failing is the
        # same class of defect as the one that made reads honour the
        # window in the first place.
        return len(self.read_all(include_expired=include_expired))

    def bytes_used(self) -> dict:
        """Storage growth, observable rather than assumed."""
        journal = self._path.stat().st_size if self._path.exists() else 0
        images = 0
        directory = self.images_dir()
        if directory.exists():
            images = sum(
                path.stat().st_size for path in directory.rglob("*") if path.is_file()
            )
        return {"journal": journal, "images": images, "total": journal + images}

    # -- delete --------------------------------------------------------

    def prune_expired(self, now: float | None = None) -> dict:
        """Drop documents older than the retention window, PIXELS INCLUDED.

        Returns the same report shape as `purge`, and for the same reason:
        an earlier version returned a bare count and deleted only the
        JOURNAL RECORDS, leaving each expired document's page images on
        disk. Retention then said the document was gone while the picture
        of it remained -- a retention window that does not remove the data
        is not a retention window, and a caller receiving only a count had
        no way to find out.

        No retention window means no pruning. The window is a constructor
        argument precisely so that "forever" has to be chosen rather than
        inherited.
        """
        if self._retention_seconds is None:
            return {
                "documents_removed": 0,
                "images_removed": 0,
                "images_retained": [],
                "complete": True,
            }

        now = self._clock() if now is None else now
        cutoff = now - self._retention_seconds

        def expired(record: dict) -> bool:
            # The negation of the read predicate, deliberately, rather
            # than a second inequality that happens to agree today.
            return not self._is_within_retention(record, cutoff)

        # Collect the doomed images BEFORE rewriting: once the records are
        # gone there is nothing left that names them.
        #
        # `include_expired=True`, and it is load-bearing. Reads began
        # filtering to the retention window on 2026-08-27, and a prune
        # that used the filtered read could no longer SEE the records it
        # exists to delete -- it would report `documents_removed: 0` and
        # leave every expired page image on disk. This is the maintenance
        # path the opt-out was added for.
        doomed = [
            self._directory / page.image_relpath
            for document in self.read_all(include_expired=True)
            if document.recorded_at < cutoff
            for page in document.pages
            if page.image_relpath
        ]
        removed = self._rewrite_keeping(lambda record: not expired(record))
        removed_images, retained_images = self._delete_paths(doomed)

        return {
            "documents_removed": removed,
            "images_removed": len(removed_images),
            "images_retained": retained_images,
            "complete": not retained_images,
        }

    def purge(self, document_id: str | None = None) -> dict:
        """Really delete. Reports what it could NOT delete.

        Reads with `include_expired=True` throughout, and that is not an
        optimisation. Once reads began honouring the retention window, a
        purge that used the filtered read could no longer SEE an expired
        record's page images -- so it deleted the journal row, reported
        `images_removed: 0` and `complete: true`, and left an unredacted
        photograph of what the wearer read on disk with nothing left
        anywhere that named it. A deletion path must see everything.

        A purge that cannot remove everything must never be presented as
        success -- `06-PRIVACY-DATA.md` requires real deletion, and a
        false claim of deletion is worse than an honest failure.
        """
        if document_id is None:
            # Everything, including any orphaned image an older or
            # interrupted write left behind.
            targets = (
                list(self.images_dir().rglob("*"))
                if self.images_dir().exists()
                else []
            )
            removed = self._rewrite_keeping(lambda record: False)
        else:
            # The images belonging to the document being REMOVED. Gathered
            # before the rewrite, because afterwards nothing names them.
            targets = [
                self._directory / page.image_relpath
                for document in self.read_all(include_expired=True)
                if document.document_id == document_id
                for page in document.pages
                if page.image_relpath
            ]
            removed = self._rewrite_keeping(
                lambda record: record.get("document_id") != document_id
            )

        removed_images, retained_images = self._delete_paths(targets)

        return {
            "documents_removed": removed,
            "images_removed": len(removed_images),
            "images_retained": retained_images,
            "complete": not retained_images,
        }

    def _delete_paths(self, paths) -> tuple[list[str], list[str]]:
        """Really delete, and report what would not go.

        Deepest first, so a directory is emptied before it is removed.
        Never raises: a locked file must be REPORTED, not thrown, because
        the caller needs the rest of the deletion to proceed and needs to
        learn that this one did not.
        """
        removed, retained = [], []
        for path in sorted(paths, key=lambda p: len(p.parts), reverse=True):
            try:
                if path.is_dir():
                    path.rmdir()
                elif path.exists():
                    path.unlink()
                removed.append(str(path))
            except OSError as exc:
                logger.warning("document store: could not remove %s: %s", path, exc)
                retained.append(str(path))
        return removed, retained

    def _rewrite_keeping(self, keep) -> int:
        """Atomically rewrite the journal, returning how many were dropped."""
        import json
        import os

        with self._lock:
            raw_records, _ = _read_raw_with_retry(self._path)
            kept = [record for record in raw_records if keep(record)]
            dropped = len(raw_records) - len(kept)
            if not self._path.exists():
                return 0
            if dropped == 0:
                # Nothing to drop is nothing to rewrite. The session prunes
                # after every document it records, and rewriting an
                # unchanged journal on each one was a whole-file fsync per
                # page and a rename a concurrent reader could refuse.
                return 0
            temp_path = self._path.with_name(self._path.name + TEMP_SUFFIX)
            try:
                with temp_path.open("w", encoding="utf-8") as handle:
                    for record in kept:
                        handle.write(json.dumps(record) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                replace_with_retry(temp_path, self._path)
            finally:
                temp_path.unlink(missing_ok=True)
            return dropped
