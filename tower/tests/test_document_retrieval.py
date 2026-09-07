"""Finding a document again -- and refusing, correctly, when it is not there.

The refusals matter more than the matches. A retrieval layer that guesses
turns "I have no record of reading that" into a confident wrong answer,
which is the failure mode the anti-hallucination requirement exists to
prevent. `07-PLATFORM-CONSTRAINTS.md` Core Principle 3: an observation
gap is never an observation of absence.

Every assertion is against a corpus this file wrote, so which document
contains which term is known independently of the ranking.
"""

import pytest

from tower.confidence import Confidence
from tower.document_memory.records import DocumentObservation, PageObservation
from tower.document_memory.retrieval import DocumentMemory, tokenise
from tower.document_memory.store import DocumentStore

TRANSFORMERS = (
    "Attention Is All You Need. The dominant sequence transduction models "
    "are based on complex recurrent or convolutional neural networks. We "
    "propose the Transformer, based solely on attention mechanisms."
)
DEPTH = (
    "Monocular Depth Notes. Relative depth is not metric distance. A single "
    "prediction is evidence, not truth. Scale must never be fabricated."
)
RECEIPT = (
    "HARDWARE STORE. Return policy: 30 days with receipt. Item cable ties "
    "4.99. Item tape measure 12.50. Total 17.49."
)


def _document(document_id, text, observed_at, confidence=Confidence.HIGH):
    return DocumentObservation(
        document_id=document_id,
        observed_at=observed_at,
        recorded_at=observed_at,
        observed_seconds=12.0,
        pages=(
            PageObservation(
                page_index=0,
                text=text,
                region_count=6,
                mean_region_confidence=0.9,
                confidence=confidence,
            ),
        ),
        title=text.split(".")[0],
        confidence=confidence,
        frames_considered=40,
        frames_ocred=2,
    )


NOW = 1_700_000_000.0


@pytest.fixture
def memory(tmp_path):
    store = DocumentStore(tmp_path)
    # Deliberately spread across an hour so the time queries have
    # something to discriminate.
    store.append(_document("transformers", TRANSFORMERS, NOW - 1800))
    store.append(_document("depth", DEPTH, NOW - 600))
    store.append(_document("receipt", RECEIPT, NOW - 60))
    return DocumentMemory(store)


class TestContentSearch:
    def test_it_finds_the_document_that_contains_the_term(self, memory):
        result = memory.search_text("transformer attention")

        assert result.sufficient_evidence
        assert result.matches[0].document_id == "transformers"

    def test_it_ranks_the_better_match_first(self, memory):
        """"metric distance" belongs to the depth notes, not the receipt."""
        result = memory.search_text("metric distance scale")

        assert result.matches[0].document_id == "depth"

    def test_a_distinctive_term_beats_a_common_one(self, memory):
        """IDF must work on a tiny corpus.

        The textbook BM25 IDF goes NEGATIVE for a term appearing in most
        documents, which would rank a document DOWN for containing the
        word that was asked for.
        """
        result = memory.search_text("receipt")

        assert result.sufficient_evidence
        assert result.matches[0].document_id == "receipt"
        assert all(match.score > 0 for match in result.matches)

    def test_every_match_reports_what_it_matched_on(self, memory):
        """A score alone asks the reader to trust the ranking."""
        result = memory.search_text("transformer attention")
        match = result.matches[0]

        assert set(match.matched_terms) == {"transformer", "attention"}
        assert match.match_kind == "lexical"

    def test_every_match_carries_a_snippet_of_captured_text(self, memory):
        """So an answer is traceable to text that was actually captured."""
        match = memory.search_text("transformer").matches[0]

        assert match.snippet
        assert "transformer" in match.snippet.lower()

    def test_the_snippet_comes_from_the_stored_text(self, memory):
        match = memory.search_text("convolutional").matches[0]
        stripped = match.snippet.replace("...", "").strip()

        assert stripped in match.document.text


class TestItRefusesRatherThanGuessing:
    def test_a_term_no_document_contains_returns_insufficient_evidence(self, memory):
        result = memory.search_text("fire extinguisher")

        assert result.sufficient_evidence is False
        assert result.matches == ()

    def test_the_refusal_says_it_is_about_the_record_not_the_world(self, memory):
        """"The paper doesn't mention X" is not assertable from a gap."""
        result = memory.search_text("fire extinguisher")

        assert "what was captured" in result.reason
        assert result.searched_documents == 3

    def test_an_empty_store_refuses_and_says_why(self, tmp_path):
        result = DocumentMemory(DocumentStore(tmp_path)).search_text("anything")

        assert result.sufficient_evidence is False
        assert "no documents" in result.reason
        assert result.searched_documents == 0

    def test_a_query_of_only_stopwords_refuses(self, memory):
        result = memory.search_text("the and of it")

        assert result.sufficient_evidence is False
        assert "no searchable terms" in result.reason

    def test_an_empty_query_refuses(self, memory):
        assert memory.search_text("").sufficient_evidence is False

    def test_the_threshold_is_reported_with_the_result(self, memory):
        """A count is meaningless without the threshold that produced it."""
        result = memory.search_text("transformer")

        assert result.min_score > 0


class TestTimeQueries:
    def test_about_thirty_minutes_ago_finds_the_right_document(self, memory):
        documents = memory.around(NOW - 1800, window_seconds=300)

        assert [document.document_id for document in documents] == ["transformers"]

    def test_the_nearest_document_comes_first(self, memory):
        """"About" is the operative word: nearest answers the question."""
        documents = memory.around(NOW - 700, window_seconds=3600)

        assert documents[0].document_id == "depth"

    def test_a_window_with_nothing_in_it_returns_nothing(self, memory):
        assert memory.around(NOW - 100_000, window_seconds=60) == []

    def test_recent_returns_newest_first(self, memory):
        documents = memory.recent(3)

        assert [document.document_id for document in documents] == [
            "receipt",
            "depth",
            "transformers",
        ]

    def test_recent_honours_its_limit(self, memory):
        assert len(memory.recent(2)) == 2


class TestCoverage:
    def test_it_reports_what_is_known_and_refuses_to_invent_a_denominator(
        self, memory
    ):
        """The system cannot see pages it was never shown.

        Reporting "2 of 2 pages" would turn an observation gap into a
        claim of completeness.
        """
        coverage = memory.coverage("transformers")

        assert coverage["pages_observed"] == 1
        assert coverage["pages_total"] is None
        assert "cannot see pages it was never shown" in coverage["pages_total_note"]

    def test_it_names_the_pages_that_were_read_badly(self, tmp_path):
        store = DocumentStore(tmp_path)
        store.append(_document("weak", DEPTH, NOW, confidence=Confidence.LOW))

        coverage = DocumentMemory(store).coverage("weak")

        assert coverage["low_confidence_pages"] == [0]

    def test_an_unknown_document_returns_none_rather_than_an_empty_shape(
        self, memory
    ):
        assert memory.coverage("never-seen") is None


class TestTokenisation:
    def test_stopwords_are_dropped(self):
        assert tokenise("the transformer and the attention") == [
            "transformer",
            "attention",
        ]

    def test_case_and_punctuation_are_ignored(self):
        assert tokenise("Transformer, ATTENTION!") == ["transformer", "attention"]

    def test_numbers_survive(self):
        """A receipt total is a searchable term."""
        assert "1749" in tokenise("Total 1749")


class TestDocumentFrequencyIsCorpusWideAndComputedOnce:
    """Document frequency is a property of (corpus, term), not of a document.

    It was computed INSIDE the per-document scoring loop, rescanning every
    document and rebuilding `set(token_list)` for each one on every call --
    O(D^2) in library size. Measured before the change: 25 documents 7.6 ms,
    800 documents 9,532 ms.

    These tests pin the two things a hoist must not break: the arithmetic,
    and the fact that a term absent from a document contributes nothing.
    """

    def _reference_page_frequency(self, corpus, term):
        """The original expression, kept as the thing parity is checked against."""
        return sum(1 for page in corpus.pages if term in set(page.tokens))

    def test_the_precomputed_frequency_equals_the_original_expression(self):
        from tower.document_memory.retrieval import _Corpus

        documents = [
            _document("transformers", TRANSFORMERS, NOW - 1800),
            _document("depth", DEPTH, NOW - 600),
            _document("receipt", RECEIPT, NOW - 60),
        ]
        corpus = _Corpus(documents)

        every_term = {token for page in corpus.pages for token in page.tokens}
        assert every_term, "the fixture corpus tokenised to nothing"

        for term in sorted(every_term):
            assert corpus.page_frequency[term] == (
                self._reference_page_frequency(corpus, term)
            ), term

    def test_a_term_in_no_document_has_frequency_zero(self):
        from tower.document_memory.retrieval import _Corpus

        corpus = _Corpus([_document("depth", DEPTH, NOW)])

        assert corpus.page_frequency.get("nonexistentterm", 0) == 0

    def test_ranking_order_survives_page_scoring(self, memory):
        """The ORDER, pinned. The scores moved on 2026-09-07, deliberately.

        Scoring became per page with the title weighted double, so the
        absolute figures the pre-hoist code produced no longer apply.
        What must not move is which document a four-term query ranks
        first: the one that contains the most of them.
        """
        result = memory.search_text("transformer attention depth receipt")

        ranked = [match.document_id for match in result.matches]
        assert ranked == ["transformers", "depth", "receipt"]
        assert all(match.page_index is not None for match in result.matches)
