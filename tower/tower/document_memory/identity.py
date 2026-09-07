"""Is this the page I already have? Two witnesses, and both must agree.

A person may stare at one page for twenty seconds, look away, and look
back. Document Memory must not remember that as three documents. It
must also not merge two different invoices printed on the same template,
or two pages of the same book, into one -- a false merge destroys a
page, a false split costs a duplicate row. So the asymmetry below is
deliberate: MERGE needs the text AND the picture to agree; anything
weaker is kept separate.

The two witnesses, measured 2026-09-06 on rendered pages under shift,
scale, perspective, exposure, JPEG and blur
(`Glasses-scratch/document-memory-v1/research/06-capture-dedup-research.md`):

- **Token-set Jaccard of the OCR text**: 0.91-1.00 for readable
  re-views of one page, 0.40-0.45 for the hardest different pages (a
  half-shared page; two invoices on one template), 0.14 for unrelated
  prose. `SAME_PAGE_TOKEN_OVERLAP` sits at 0.70, above the hardest
  negative with room for real OCR noise.
- **A 64-bit perceptual hash of the text region**: 0-2 bits apart for
  the same page under every transform, 8-24 for different prose. But 2
  bits for two invoices on one template -- the hash sees layout, not
  words -- which is why it is necessary and never sufficient.

Numeric tokens get their own vote. Two invoices share every word except
the numbers, and the numbers are the document.
"""

import re
from dataclasses import dataclass

import cv2
import numpy as np

# Text-set overlap at or above which two readings are the same words.
SAME_PAGE_TOKEN_OVERLAP = 0.70
# Below this, unrelated. Between the two, "shares text" -- kept separate.
RELATED_TOKEN_OVERLAP = 0.25
# Hamming distance (of 64 bits) at or below which two regions look alike.
SAME_LOOK_MAX_DISTANCE = 6
# Numeric tokens must agree this well when there are enough to matter.
NUMERIC_TOKEN_OVERLAP = 0.5
MIN_NUMERIC_TOKENS = 3
# A reading with fewer words than this cannot testify either way.
MIN_WORDS_FOR_TEXT = 3
# And one with a mean confidence below this is too uncertain to split on.
MIN_CONFIDENCE_TO_TESTIFY = 0.30

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMERIC = re.compile(r"\d+(?:[.,]\d+)?")

HASH_SIZE = 8
HASH_DCT_SIZE = 32


def tokenise(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def token_overlap(left: str, right: str) -> float:
    """Jaccard overlap of token SETS.

    Sets, not counts: a page re-observed produces the same vocabulary but
    not the same word frequencies, because OCR splits and merges words
    differently on each view.
    """
    a, b = set(tokenise(left)), set(tokenise(right))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(left: str, right: str) -> float:
    """How much of the SMALLER vocabulary the larger one contains.

    A partial view of a page carries a subset of its words; Jaccard
    punishes that, containment does not.
    """
    a, b = set(tokenise(left)), set(tokenise(right))
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def numeric_tokens(text: str) -> set[str]:
    return set(_NUMERIC.findall(text))


def numbers_agree(left: str, right: str) -> bool:
    """True unless both sides carry enough numbers and they differ."""
    a, b = numeric_tokens(left), numeric_tokens(right)
    if len(a) < MIN_NUMERIC_TOKENS or len(b) < MIN_NUMERIC_TOKENS:
        return True
    return len(a & b) / len(a | b) >= NUMERIC_TOKEN_OVERLAP


def perceptual_hash(gray: np.ndarray) -> str | None:
    """pHash: a 32x32 DCT, the low 8x8 minus DC, thresholded at the median.

    Sixteen hex characters. Not imagery -- nothing renders back from it
    -- and not a fingerprint of the words either, which is why it is
    only ever one of two witnesses.
    """
    if gray is None or gray.size == 0:
        return None
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(
        gray, (HASH_DCT_SIZE, HASH_DCT_SIZE), interpolation=cv2.INTER_AREA
    )
    dct = cv2.dct(np.float32(small))
    low = dct[:HASH_SIZE, :HASH_SIZE].flatten()
    # Drop the DC term: it is mean brightness, which is the lamp.
    low = low[1:]
    threshold = float(np.median(low))
    bits = 0
    for value in low:
        bits = (bits << 1) | int(value > threshold)
    # 63 bits from an 8x8 block minus DC; pad to a 64-bit word.
    return f"{bits:016x}"


def hash_distance(left: str | None, right: str | None) -> int | None:
    if not left or not right:
        return None
    try:
        return bin(int(left, 16) ^ int(right, 16)).count("1")
    except ValueError:
        return None


@dataclass(frozen=True)
class Reading:
    """The evidence one page offers for an identity decision."""

    text: str
    mean_confidence: float | None
    visual_hash: str | None

    @property
    def word_count(self) -> int:
        return len(tokenise(self.text))

    @property
    def can_testify(self) -> bool:
        """Readable enough that its words mean something."""
        return (
            self.word_count >= MIN_WORDS_FOR_TEXT
            and (self.mean_confidence or 0.0) >= MIN_CONFIDENCE_TO_TESTIFY
        )


@dataclass(frozen=True)
class Verdict:
    """What the two witnesses said, and what was decided."""

    decision: str  # "same" | "shares-text" | "same-look" | "different" | "abstain"
    text_overlap: float
    text_containment: float
    hash_distance: int | None
    numbers_agree: bool

    @property
    def same(self) -> bool:
        return self.decision == "same"


def compare(existing: Reading, incoming: Reading) -> Verdict:
    """Decide whether `incoming` is a re-sighting of `existing`.

    "same" requires the words to agree (overlap or, for a partial view,
    containment), the numbers not to disagree, AND the regions to look
    alike when both hashes exist. When only one hash exists, the words
    alone may carry a strong overlap; when neither reading can testify,
    the answer is "abstain" and the caller keeps them separate.
    """
    distance = hash_distance(existing.visual_hash, incoming.visual_hash)
    if not (existing.can_testify and incoming.can_testify):
        return Verdict("abstain", 0.0, 0.0, distance, True)

    overlap = token_overlap(existing.text, incoming.text)
    contained = containment(existing.text, incoming.text)
    numbers = numbers_agree(existing.text, incoming.text)
    words_same = (
        overlap >= SAME_PAGE_TOKEN_OVERLAP
        or (contained >= 0.85 and overlap >= RELATED_TOKEN_OVERLAP)
    ) and numbers
    looks_same = distance is not None and distance <= SAME_LOOK_MAX_DISTANCE

    if words_same and (looks_same or distance is None):
        decision = "same"
    elif words_same:
        # The words agree and the picture does not: a re-render of the
        # same text somewhere else, or a hash thrown by a partial view.
        # Not merged. A false merge destroys a page.
        decision = "shares-text"
    elif looks_same:
        decision = "same-look"
    elif overlap >= RELATED_TOKEN_OVERLAP or contained >= 0.60:
        decision = "shares-text"
    else:
        decision = "different"
    return Verdict(decision, overlap, contained, distance, numbers)
