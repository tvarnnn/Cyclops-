"""Finding a document again, and refusing when it cannot be found.

Three questions, matching what a wearer actually asks:

- *"about thirty minutes ago"*  -> a time window;
- *"the one about transformers"* -> content;
- *"what have I read lately"*    -> recent history.

**The content search is LEXICAL, and this module says so in its own
output.** BM25 matches a document containing the word "transformer"; it
does not match a paraphrase that never uses it. Calling that "semantic
retrieval" would be an overclaim of exactly the kind `02-DEVELOPMENT-RULES.md`
Rule 16 exists to prevent, so every result carries `match_kind:
"lexical"` and the API name says `search_text`, not `search_meaning`.

Three things changed on 2026-09-07, all measured against OCR output
rather than clean text:

**Pages are the unit of scoring.** A document is a dwell; a dwell may
hold several pages; the wearer asks about the page. Scoring pages and
reporting the best one per document is what lets a result say WHICH
page said "port 8000", and the snippet comes from that page.

**A query term matches an OCR'd token that is nearly it.** OCR reads
"Kubernetes" as "Kubemetes" often enough that exact tokens miss real
pages. A term of five or more characters also matches a token within
one edit of it, or one that begins with it; shorter terms match exactly.
Tolerance is bounded so "port" does not match "sport".

**The corpus is cached on the journal's stamp.** The status producer
already stat-gates on `(mtime_ns, size)`; the corpus does the same, so a
query re-tokenises the library only when the library changed. A newly
persisted page changes the stamp and is searchable on the next query,
which is the "immediately searchable" the product needs.

Embeddings are the documented upgrade path, with a named trigger: when a
measured query set shows lexical recall failing on paraphrase. Until then
BM25 needs no dependency and -- more important here than ranking
finesse -- is **explainable**: every result carries the terms it matched
on and a snippet of the text it matched in, so an answer is always
traceable back to text that was actually captured.

Nothing here ever synthesises an answer. It returns records, or it
returns nothing and says so.
"""

import math
import re
import threading
from dataclasses import dataclass, field

from tower.confidence import Confidence
from tower.document_memory.records import DocumentObservation

# Standard BM25 constants. k1 controls term-frequency saturation, b the
# length normalisation. These are the widely used defaults and there is no
# corpus here to tune them against -- tuning on three documents would be
# fitting noise.
BM25_K1 = 1.5
BM25_B = 0.75

# Below this a match is noise. Reported alongside every result so the
# threshold is visible rather than buried, and so it can be moved from
# data once a real query set exists.
MIN_SCORE = 0.10

# A term this long or longer may match a token one edit away or one it
# is a prefix of. Shorter terms match exactly: "port" must not find
# "sport", and a four-letter word one edit from another four-letter
# word is usually a different word.
FUZZY_MIN_TERM_LENGTH = 5
# And a fuzzy match is worth less than an exact one, so a page that
# actually says the word outranks one that nearly does.
FUZZY_WEIGHT = 0.6

# Words in a page's TITLE count this many times over. The title is the
# document's first line, which is what a person remembers.
TITLE_WEIGHT = 2

_TOKEN = re.compile(r"[a-z0-9]+")

# Deliberately tiny. A large stopword list is a language model in
# disguise; these are the words that would otherwise dominate every score
# in English prose.
STOPWORDS = frozenset(
    """a an and are as at be by for from has have in is it its of on or
    that the this to was were will with""".split()
)


def tokenise(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


# The confusions a Latin-script recogniser makes most, folded to one
# form before the edit-distance test. "rn" read as "m" is two edits and
# the single most common OCR error in body text; "vv" as "w" the next.
# Applied only on the fuzzy path, to terms long enough to use it.
_OCR_FOLDS = (("rn", "m"), ("vv", "w"), ("ii", "u"))


def ocr_fold(token: str) -> str:
    for source, target in _OCR_FOLDS:
        token = token.replace(source, target)
    if any(character.isalpha() for character in token):
        token = token.replace("0", "o").replace("1", "l")
    return token


def within_one_edit(left: str, right: str) -> bool:
    """Levenshtein distance <= 1, without building the matrix.

    Two strings of equal length differ by one substitution; strings one
    apart in length differ by one insertion. Anything else is farther.
    """
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(1 for a, b in zip(left, right) if a != b) == 1
    short, long_ = (left, right) if len(left) < len(right) else (right, left)
    index = 0
    while index < len(short) and short[index] == long_[index]:
        index += 1
    return short[index:] == long_[index + 1 :]


@dataclass(frozen=True)
class Match:
    """One retrieved document, and why it was retrieved."""

    document: DocumentObservation
    score: float
    match_kind: str
    matched_terms: tuple[str, ...] = ()
    snippet: str = ""
    # Which page carried the best-scoring text, and its own score.
    page_index: int | None = None
    page_score: float = 0.0
    # Whether any matched term was a near-miss rather than the word.
    fuzzy: bool = False
    # How many query terms this page contains EXACTLY. The primary sort
    # key: a page that says the word outranks one that nearly says it,
    # whatever BM25's length normalisation makes of their lengths.
    exact_terms: int = 0

    @property
    def document_id(self) -> str:
        return self.document.document_id

    def to_json_dict(self) -> dict:
        return {
            "document_id": self.document.document_id,
            "title": self.document.title,
            "observed_at": self.document.observed_at,
            "time_basis": self.document.time_basis,
            "observed_seconds": self.document.observed_seconds,
            "pages_observed": self.document.pages_observed,
            "confidence": self.document.confidence.value,
            "score": round(self.score, 4),
            "match_kind": self.match_kind,
            "matched_terms": list(self.matched_terms),
            "snippet": self.snippet,
            "page_index": self.page_index,
            "fuzzy": self.fuzzy,
        }


@dataclass(frozen=True)
class QueryResult:
    """Matches, or an explicit refusal. Never a fabrication.

    `sufficient_evidence` is the field that matters. A query with no
    confident match returns `False` and an empty `matches`, and a caller
    that renders that as "I don't have a record of reading that" is
    telling the truth. Rendering a best guess instead would not be.
    """

    query: str
    matches: tuple[Match, ...] = ()
    sufficient_evidence: bool = False
    reason: str = ""
    searched_documents: int = 0
    searched_pages: int = 0
    min_score: float = MIN_SCORE

    def to_json_dict(self) -> dict:
        return {
            "query": self.query,
            "sufficient_evidence": self.sufficient_evidence,
            "reason": self.reason,
            "searched_documents": self.searched_documents,
            "searched_pages": self.searched_pages,
            "min_score": self.min_score,
            "matches": [match.to_json_dict() for match in self.matches],
        }


@dataclass
class _Page:
    """One scoring unit: a page's tokens plus its document's title's."""

    document_index: int
    page_index: int
    tokens: list[str]
    text: str
    frequencies: dict = field(default_factory=dict)

    def __post_init__(self):
        for token in self.tokens:
            self.frequencies[token] = self.frequencies.get(token, 0) + 1


@dataclass
class _Corpus:
    documents: list[DocumentObservation]
    pages: list[_Page] = field(default_factory=list)

    def __post_init__(self):
        for document_index, document in enumerate(self.documents):
            title_tokens = tokenise(document.title or "") * TITLE_WEIGHT
            readable = [page for page in document.pages if page.text.strip()]
            if not readable:
                continue
            for page in readable:
                self.pages.append(
                    _Page(
                        document_index=document_index,
                        page_index=page.page_index,
                        tokens=tokenise(page.text) + title_tokens,
                        text=page.text,
                    )
                )
        lengths = [len(page.tokens) for page in self.pages]
        self.average_length = sum(lengths) / len(lengths) if lengths else 0.0
        # How many pages contain each token. A property of (corpus,
        # term) and NOT of the page being scored, computed once: an
        # earlier version recomputed it inside the scoring loop and was
        # quadratic in library size (356.92 ms -> 14.18 ms at 800 docs
        # when hoisted).
        self.page_frequency: dict[str, int] = {}
        for page in self.pages:
            for token in set(page.tokens):
                self.page_frequency[token] = self.page_frequency.get(token, 0) + 1
        self.vocabulary = list(self.page_frequency)
        self.folded_vocabulary = [ocr_fold(token) for token in self.vocabulary]

    def expand(self, term: str) -> dict[str, float]:
        """The corpus tokens a query term matches, each with its weight.

        Exact is 1.0. For a long enough term, a token within one edit or
        one the term is a prefix of is `FUZZY_WEIGHT`. Bounded to the
        vocabulary, which is what keeps this cheap: a library of a
        thousand pages has a few thousand distinct tokens.
        """
        matches = {}
        if term in self.page_frequency:
            matches[term] = 1.0
        if len(term) < FUZZY_MIN_TERM_LENGTH:
            return matches
        folded_term = ocr_fold(term)
        for token, folded in zip(self.vocabulary, self.folded_vocabulary):
            if token == term:
                continue
            if (
                token.startswith(term)
                or within_one_edit(term, token)
                or within_one_edit(folded_term, folded)
            ):
                matches[token] = max(matches.get(token, 0.0), FUZZY_WEIGHT)
        return matches


class _CorpusCache:
    """One parsed corpus per store path, keyed on the journal's stamp.

    Process-wide and lock-guarded, because the routes construct a fresh
    `DocumentStore` per request and a cache on the store would be a
    cache of one.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict = {}

    def get(self, store) -> _Corpus:
        path = getattr(store, "path", None)
        if path is None:
            return _Corpus(store.read_all())
        try:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size, store.retention_seconds)
        except FileNotFoundError:
            stamp = None
        key = str(path)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached[0] == stamp and stamp is not None:
                return cached[1]
        corpus = _Corpus(store.read_all())
        with self._lock:
            self._entries[key] = (stamp, corpus)
        return corpus


_CACHE = _CorpusCache()


class DocumentMemory:
    """The read side. Read-only by construction -- it cannot write."""

    def __init__(self, store) -> None:
        self._store = store

    def recent(self, limit: int = 10) -> list[DocumentObservation]:
        """The most recently observed documents, newest first.

        By LAST observation: a document seen again this morning is more
        recent than one seen once an hour ago, whatever their first
        observations say.
        """
        documents = self._store.read_all()
        documents.sort(key=lambda document: document.last_observed_at, reverse=True)
        return documents[:limit]

    def around(
        self, when: float, window_seconds: float = 900.0
    ) -> list[DocumentObservation]:
        """*"About thirty minutes ago"*, as a window rather than an instant.

        Ordered by how close each observation is to the asked-about time,
        because "about" is the operative word: the nearest document is
        the answer, and the window only bounds how far "about" stretches.
        A sighting inside the window counts as much as a first
        observation: the document WAS in view then.

        `observed_at` is `tower-receipt` time, not capture time. Over the
        minutes-scale windows this API is for, the difference is far below
        the resolution of the question -- but a caller rendering an exact
        clock time must still label it as received (Rule 16).
        """
        scored = []
        for document in self._store.read_all():
            times = [document.observed_at] + [s.observed_at for s in document.sightings]
            distance = min(abs(at - when) for at in times)
            if distance <= window_seconds:
                scored.append((distance, document))
        scored.sort(key=lambda entry: entry[0])
        return [document for _distance, document in scored]

    def search_text(
        self, query: str, limit: int = 5, min_score: float = MIN_SCORE
    ) -> QueryResult:
        """Lexical BM25 over stored OCR text, page by page. Refuses rather than guesses."""
        corpus = _CACHE.get(self._store)
        documents = corpus.documents
        query_terms = tokenise(query)

        if not documents:
            return QueryResult(
                query=query,
                reason="no documents have been observed",
                searched_documents=0,
                min_score=min_score,
            )
        if not query_terms:
            return QueryResult(
                query=query,
                reason="the query contains no searchable terms",
                searched_documents=len(documents),
                searched_pages=len(corpus.pages),
                min_score=min_score,
            )

        expansions = {term: corpus.expand(term) for term in set(query_terms)}
        best_per_document: dict[int, Match] = {}
        for page in corpus.pages:
            score, matched, fuzzy, exact = _bm25(expansions, corpus, page)
            if score < min_score:
                continue
            document = documents[page.document_index]
            current = best_per_document.get(page.document_index)
            if current is not None and (current.exact_terms, current.page_score) >= (
                exact,
                score,
            ):
                continue
            best_per_document[page.document_index] = Match(
                document=document,
                score=score,
                match_kind="lexical",
                matched_terms=tuple(sorted(matched)),
                snippet=_snippet(page.text, matched),
                page_index=page.page_index,
                page_score=score,
                fuzzy=fuzzy,
                exact_terms=exact,
            )

        scored = sorted(
            best_per_document.values(),
            key=lambda match: (match.exact_terms, match.score),
            reverse=True,
        )
        if not scored:
            return QueryResult(
                query=query,
                reason=(
                    "no observed document contains these terms; this is a "
                    "statement about what was captured, not about what the "
                    "documents say"
                ),
                searched_documents=len(documents),
                searched_pages=len(corpus.pages),
                min_score=min_score,
            )
        return QueryResult(
            query=query,
            matches=tuple(scored[:limit]),
            sufficient_evidence=True,
            reason="lexical match on stored OCR text",
            searched_documents=len(documents),
            searched_pages=len(corpus.pages),
            min_score=min_score,
        )

    def coverage(self, document_id: str) -> dict | None:
        """What is known about a document, and what is not.

        Deliberately reports `pages_observed` with no `pages_total`. This
        cartridge cannot know how many pages a physical document has, and
        inventing a denominator would turn "we saw two pages" into "we saw
        two of two pages" -- an observation gap presented as completeness
        (Core Principle 3).
        """
        document = self._store.read_one(document_id)
        if document is None:
            return None
        return {
            "document_id": document.document_id,
            "pages_observed": document.pages_observed,
            "pages_total": None,
            "pages_total_note": (
                "unknown: the system cannot see pages it was never shown"
            ),
            "words_captured": document.word_count,
            "observed_seconds": document.observed_seconds,
            "frames_considered": document.frames_considered,
            "frames_ocred": document.frames_ocred,
            "confidence": document.confidence.value,
            "low_confidence_pages": [
                page.page_index
                for page in document.pages
                if page.confidence in (Confidence.LOW, Confidence.UNKNOWN)
            ],
        }


def _bm25(expansions, corpus: _Corpus, page: _Page) -> tuple[float, set[str], bool, int]:
    tokens = page.tokens
    if not tokens:
        return 0.0, set(), False, 0

    length = len(tokens)
    total_pages = len(corpus.pages)
    score = 0.0
    matched = set()
    fuzzy = False
    exact = 0
    for term, candidates in expansions.items():
        # The best-weighted corpus token this term matches on this page.
        best_weight = 0.0
        best_token = None
        for token, weight in candidates.items():
            if page.frequencies.get(token, 0) and weight > best_weight:
                best_weight, best_token = weight, token
        if best_token is None:
            continue
        matched.add(best_token)
        if best_weight < 1.0:
            fuzzy = True
        else:
            exact += 1
        term_frequency = page.frequencies[best_token]
        containing = corpus.page_frequency.get(best_token, 0)
        # The +0.5/+0.5 smoothing keeps IDF positive on a tiny corpus. On
        # three pages the textbook form goes NEGATIVE for a term that
        # appears in most of them, which would rank a page DOWN for
        # containing the word that was asked for.
        idf = math.log(1.0 + (total_pages - containing + 0.5) / (containing + 0.5))
        denominator = term_frequency + BM25_K1 * (
            1 - BM25_B + BM25_B * length / max(corpus.average_length, 1.0)
        )
        score += best_weight * idf * (term_frequency * (BM25_K1 + 1)) / denominator
    return score, matched, fuzzy, exact


def _snippet(text: str, matched_terms, width: int = 160) -> str:
    """Text around the first matched term, so a match is checkable.

    A score alone asks the reader to trust the ranking. A snippet lets
    them see the words the system actually captured.
    """
    if not text:
        return ""
    lowered = text.lower()
    position = -1
    for term in matched_terms:
        found = lowered.find(term)
        if found != -1 and (position == -1 or found < position):
            position = found
    if position == -1:
        return text[:width]
    start = max(position - width // 3, 0)
    return ("..." if start else "") + text[start : start + width].strip() + "..."
