"""Search over what was captured: pages, near-misses, and freshness."""

import time

import pytest

from tower.confidence import Confidence
from tower.document_memory.records import DocumentObservation, PageObservation
from tower.document_memory.retrieval import (
    FUZZY_MIN_TERM_LENGTH,
    DocumentMemory,
    within_one_edit,
)
from tower.document_memory.store import DocumentStore

NOW = 1_700_000_000.0


def _page(index, text, confidence=0.9):
    return PageObservation(
        page_index=index,
        text=text,
        region_count=max(1, len(text.split())),
        mean_region_confidence=confidence,
        confidence=Confidence.HIGH,
        readable=bool(text.strip()),
    )


def _document(document_id, pages, at=NOW, title=None):
    return DocumentObservation(
        document_id=document_id,
        observed_at=at,
        recorded_at=at,
        observed_seconds=5.0,
        pages=tuple(_page(index, text) for index, text in enumerate(pages)),
        title=title or pages[0].split(".")[0][:60],
        confidence=Confidence.HIGH,
    )


@pytest.fixture
def store(tmp_path):
    made = DocumentStore(tmp_path)
    made.append(
        _document(
            "cluster",
            [
                "Kubernetes cluster setup. The control plane runs three nodes.",
                "Networking. Pods reach each other over the overlay network on port 8000.",
            ],
            at=NOW - 600,
        )
    )
    made.append(
        _document(
            "receipt",
            ["HARDWARE STORE. Item cable ties 4.99. Total 17.49."],
            at=NOW - 60,
        )
    )
    return made


class TestPagesAreTheUnit:
    def test_a_match_names_the_page_it_came_from(self, store):
        result = DocumentMemory(store).search_text("port 8000")

        assert result.sufficient_evidence
        match = result.matches[0]
        assert match.document_id == "cluster"
        assert match.page_index == 1
        assert "8000" in match.snippet

    def test_searched_pages_is_reported(self, store):
        result = DocumentMemory(store).search_text("nodes")

        assert result.searched_documents == 2
        assert result.searched_pages == 3

    def test_one_document_appears_once_however_many_pages_match(self, store):
        result = DocumentMemory(store).search_text("network overlay pods plane")

        assert [match.document_id for match in result.matches].count("cluster") == 1


class TestNearMisses:
    def test_an_ocr_typo_still_finds_the_page(self, store):
        """OCR reads 'Kubernetes' as 'Kubemetes'. The wearer types the word."""
        store.append(_document("typo", ["Kubemetes upgrade notes for the staging ring."], at=NOW - 30))

        result = DocumentMemory(store).search_text("kubernetes")

        ids = [match.document_id for match in result.matches]
        assert "cluster" in ids and "typo" in ids
        typo = next(match for match in result.matches if match.document_id == "typo")
        assert typo.fuzzy is True
        assert "kubemetes" in typo.matched_terms

    def test_an_exact_match_outranks_a_near_one(self, store):
        store.append(_document("typo", ["Kubemetes upgrade notes for the staging ring."], at=NOW - 30))

        result = DocumentMemory(store).search_text("kubernetes")

        assert result.matches[0].document_id == "cluster"
        assert result.matches[0].fuzzy is False

    def test_short_terms_match_exactly_only(self, store):
        store.append(_document("sport", ["The sport shop sells nets and racquets."], at=NOW - 30))

        result = DocumentMemory(store).search_text("port")

        assert [match.document_id for match in result.matches] == ["cluster"]

    def test_a_prefix_matches_a_longer_token(self, store):
        result = DocumentMemory(store).search_text("kubern")

        assert result.matches[0].document_id == "cluster"
        assert len("kubern") >= FUZZY_MIN_TERM_LENGTH

    def test_the_rn_to_m_confusion_is_folded(self):
        from tower.document_memory.retrieval import ocr_fold

        # Two edits apart as written; one form once folded.
        assert not within_one_edit("kubernetes", "kubemetes")
        assert ocr_fold("kubernetes") == ocr_fold("kubemetes")

    @pytest.mark.parametrize(
        "left,right,expected",
        [
            ("kubernetes", "kubemetes", False),  # two edits: rn -> m is a substitution plus a deletion
            ("kubernetes", "kubernetez", True),
            ("kubernetes", "kubernete", True),
            ("kubernetes", "kubernetesx", True),
            ("port", "sport", True),
            ("port", "ports", True),
            ("abc", "xyz", False),
        ],
    )
    def test_within_one_edit(self, left, right, expected):
        assert within_one_edit(left, right) is expected


class TestFreshness:
    def test_a_newly_appended_page_is_searchable_on_the_next_query(self, store):
        memory = DocumentMemory(store)
        assert not memory.search_text("lighthouse").sufficient_evidence

        # Same second, possibly same mtime tick: the stamp includes size.
        store.append(_document("new", ["The lighthouse keeper's log for March."], at=NOW))

        result = memory.search_text("lighthouse")
        assert result.sufficient_evidence
        assert result.matches[0].document_id == "new"

    def test_a_purged_page_stops_matching(self, store):
        memory = DocumentMemory(store)
        assert memory.search_text("receipt cable").sufficient_evidence

        store.purge("receipt")
        time.sleep(0.01)

        assert not memory.search_text("cable ties").sufficient_evidence

    def test_recent_orders_by_last_sighting(self, store):
        from dataclasses import replace

        from tower.document_memory.records import Sighting

        old = store.read_one("cluster")
        store.update(replace(old, sightings=(Sighting(observed_at=NOW + 5, observed_seconds=3.0),)))

        recent = DocumentMemory(store).recent()

        assert [document.document_id for document in recent] == ["cluster", "receipt"]

    def test_around_counts_a_sighting_as_being_in_view(self, store):
        from dataclasses import replace

        from tower.document_memory.records import Sighting

        old = store.read_one("cluster")
        store.update(replace(old, sightings=(Sighting(observed_at=NOW + 3600, observed_seconds=3.0),)))

        around = DocumentMemory(store).around(NOW + 3600, window_seconds=60)

        assert [document.document_id for document in around] == ["cluster"]


class TestRefusals:
    def test_unreadable_pages_are_not_searched(self, store):
        store.append(
            DocumentObservation(
                document_id="blank",
                observed_at=NOW,
                recorded_at=NOW,
                observed_seconds=2.0,
                pages=(PageObservation(page_index=0, text="", region_count=4, readable=False),),
            )
        )

        result = DocumentMemory(store).search_text("nodes")

        assert result.searched_documents == 3
        assert result.searched_pages == 3
