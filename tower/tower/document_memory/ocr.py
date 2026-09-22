"""Turning a corrected page image into text, behind a seam.

A seam rather than a direct call for two reasons, both practical.

**Cost.** EasyOCR takes ~5 s to construct a reader and ~1.2 s per page on
this CPU. A default test suite that paid that per assertion would be
unusable, and one that downloaded a model would fail on a train. Tests
use `FixedTextRecogniser`; the real engine is exercised by an opt-in
integration test.

**Substitutability.** `docs/superpowers/research/2026-08-20-document-memory-design.md`
surveyed the OCR field and deliberately did not select one, because the
choice should follow measurement. The seam is how a later measurement can
change the answer without touching the pipeline.

Everything here is **model inference, not measured fact**
(`07-PLATFORM-CONSTRAINTS.md` Core Principle 2). Per-region confidence is
carried through so a low-confidence reading cannot silently become a
confident answer.
"""

import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from tower.confidence import Confidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextRegion:
    """One recognised run of text, and where on the page it sat.

    The box is not decoration. A recogniser splits a heading into several
    regions -- "Attention", "Is All You Need" -- so reconstructing a title
    from the first region alone yields one word. Grouping regions that
    share a line needs their vertical position, and only the recogniser
    knows it.
    """

    text: str
    confidence: float
    box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1

    @property
    def y_centre(self) -> float | None:
        if self.box is None:
            return None
        return (self.box[1] + self.box[3]) / 2.0

    @property
    def height(self) -> float | None:
        if self.box is None:
            return None
        return abs(self.box[3] - self.box[1])


@dataclass(frozen=True)
class OcrResult:
    """What OCR actually returned, with how sure it was.

    An empty result is a real answer -- "we looked and found no readable
    text" -- and is stored as such. It is not the same as never looking,
    and Core Principle 3 forbids collapsing the two.
    """

    text: str
    regions: tuple[TextRegion, ...] = ()

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def mean_confidence(self) -> float | None:
        if not self.regions:
            return None
        return sum(region.confidence for region in self.regions) / len(self.regions)

    @property
    def min_confidence(self) -> float | None:
        if not self.regions:
            return None
        return min(region.confidence for region in self.regions)

    @property
    def confidence_label(self) -> Confidence:
        """A LABEL, never a stored score.

        Stored as a label so a later threshold change cannot silently
        relabel history (Rule 16). Derived from the MEAN rather than the
        minimum: one hard word in a paragraph should not condemn the page.
        """
        return Confidence.from_score(self.mean_confidence)


@runtime_checkable
class TextRecogniser(Protocol):
    """Anything that can read a corrected page image."""

    name: str

    def read(self, page_gray) -> OcrResult: ...

    def release(self) -> None: ...


class FixedTextRecogniser:
    """Returns text the test chose. The default suite's recogniser.

    Not a mock of EasyOCR's behaviour -- a substitute for it. Tests that
    care about OCR ACCURACY use the real engine behind an opt-in marker;
    tests that care about the pipeline use this, and are therefore fast,
    offline and deterministic.
    """

    name = "fixed"

    def __init__(self, pages=None, confidence: float = 0.9) -> None:
        # Each entry is one page's text. Newlines separate REGIONS, which
        # is the shape a real recogniser returns -- one entry per detected
        # line, joined with spaces into `text`. A fake that returned one
        # undivided blob would let the pipeline depend on structure the
        # real engine never provides.
        self._pages = list(pages or [])
        self._confidence = confidence
        self.calls = 0

    def read(self, page_gray) -> OcrResult:
        self.calls += 1
        if not self._pages:
            return OcrResult(text="")
        raw = self._pages[min(self.calls - 1, len(self._pages) - 1)]
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        # Synthetic boxes, one line per row, so title grouping is
        # exercised by the fake exactly as it is by the real engine.
        regions = tuple(
            TextRegion(
                text=line,
                confidence=self._confidence,
                box=(0.0, index * 40.0, 400.0, index * 40.0 + 30.0),
            )
            for index, line in enumerate(lines)
        )
        return OcrResult(
            text=" ".join(region.text for region in regions), regions=regions
        )

    def detect(self, gray):
        """Where the text is, by morphology. See `gate.classical_text_boxes`.

        The fake reads text a test chose; it cannot know where a test's
        rendered page is. The classical detector can, deterministically
        and offline, on the rendered pages the fixtures produce.
        """
        from tower.document_memory.gate import classical_text_boxes

        return classical_text_boxes(gray)

    def release(self) -> None:
        return None


class OcrExtraMissing(RuntimeError):
    """The `[ocr]` extra is not installed in this environment.

    Its own type so a message written for a person passes through
    `client_safe_reason` to the phone, where "ModuleNotFoundError" would
    not have told anyone what to install.
    """


def require_ocr_extra() -> None:
    """Refuse, with instructions, when easyocr is not installed.

    `find_spec` locates without importing: the web process must not pay
    torch's import to learn whether a cartridge is available, and
    `test_the_ocr_dependency_is_not_imported_at_module_load` holds it to
    that.
    """
    import importlib.util

    for name in ("torch", "easyocr"):
        try:
            present = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            # A finder that refuses the name -- the torchless-host tests
            # install one -- is a host without it.
            present = False
        if not present:
            raise OcrExtraMissing(
                f"{name} is not installed; the OCR extra needs the `[ml]` "
                "torch wheels first, then `pip install -e .[ocr]`"
            )


def resolve_ocr_device(requested: str) -> bool:
    """Whether the reader may use CUDA. "auto" downgrades; "cuda" refuses.

    Returns a bool because that is what EasyOCR takes. The rule is the
    one `cartridge_runtime._resolve_device` states for Scene: an
    unnoticed downgrade from cuda to cpu turns a GPU deployment into a
    CPU one with a GPU label on it, which is worse than a failure.
    """
    if requested == "cpu":
        return False
    import torch

    available = bool(torch.cuda.is_available())
    if requested == "cuda" and not available:
        raise RuntimeError("cuda requested but torch reports it is unavailable")
    return available


@dataclass(frozen=True)
class TextBox:
    """One region the DETECTOR found, before any recognition.

    Axis-aligned, in the coordinates of the image it was found in. The
    detector answers "is there text, and where" for a few tens of
    milliseconds on a GPU; recognition, ten times the cost, is spent
    only on a frame this evidence has earned.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return abs(self.y1 - self.y0)

    @property
    def area(self) -> float:
        return abs(self.x1 - self.x0) * abs(self.y1 - self.y0)


class EasyOcrRecogniser:
    """EasyOCR, loaded explicitly and released for real.

    Chosen over the alternatives for one disqualifying reason on each of
    them: `pytesseract` needs a system binary that is not present and pip
    cannot install; `rapidocr_onnxruntime` and PaddleOCR pull
    `opencv-python` alongside this project's `opencv-python-headless`,
    and two cv2 distributions in one environment is a known breakage.
    EasyOCR adds no cv2 and reuses the torch already in the `ml` extra.

    Measured 2026-09-06 on an RTX 5070 against rendered pages with known
    text: reader construction 1.3 s, 0.27 s per 800x1040 page on CUDA
    against 1.86 s on CPU, 0.98 word recall at 360x640 when the page
    fills the frame. The detector-only stage (`detect`) is 31-47 ms at
    360x640 and 55-66 ms at 504x896, and 0 boxes on a noise frame.

    `device` follows the Tower's vocabulary ("auto" / "cuda" / "cpu")
    and is resolved at `load`, on the worker thread, so the web process
    never imports torch to construct a session.
    """

    name = "easyocr"

    def __init__(self, languages=("en",), device: str = "auto") -> None:
        self._languages = list(languages)
        self._device = device
        self._gpu: bool | None = None
        self._reader = None

    @property
    def device(self) -> str | None:
        """The device actually in use once loaded, else None."""
        if self._gpu is None:
            return None
        return "cuda" if self._gpu else "cpu"

    def load(self) -> None:
        # Local import: easyocr is an optional [ocr] extra and nothing
        # outside a document-reading path may require it.
        import easyocr

        self._gpu = resolve_ocr_device(self._device)
        self._reader = easyocr.Reader(
            self._languages, gpu=self._gpu, verbose=False
        )

    def detect(self, gray) -> tuple[TextBox, ...]:
        """Where the text is, without reading it. The cheap GPU stage."""
        if self._reader is None:
            self.load()
        horizontal, free = self._reader.detect(gray)
        boxes = []
        for group in horizontal:
            for entry in group:
                try:
                    x0, x1, y0, y1 = (float(value) for value in entry)
                except (TypeError, ValueError):
                    continue
                boxes.append(TextBox(x0, y0, x1, y1))
        for group in free:
            for polygon in group:
                bounds = _bounds(polygon)
                if bounds is not None:
                    boxes.append(TextBox(*bounds))
        return tuple(boxes)

    def read(self, page_gray) -> OcrResult:
        if self._reader is None:
            self.load()
        raw = self._reader.readtext(page_gray)
        regions = tuple(
            TextRegion(
                text=str(entry[1]).strip(),
                confidence=float(entry[2]),
                box=_bounds(entry[0]),
            )
            for entry in raw
            if str(entry[1]).strip()
        )
        return OcrResult(
            text=" ".join(region.text for region in regions), regions=regions
        )

    def release(self) -> None:
        """Drop the models and hand the GPU memory back.

        Dropping the reference alone left ~1.4 GB reserved on the device
        after a session (measured); the cache release brings it to the
        ~0.2 GB the CUDA context itself costs, which is what "a cartridge
        that is not running holds nothing worth mentioning" has to mean
        on a Tower that will run other cartridges next.
        """
        import gc

        self._reader = None
        was_gpu = self._gpu
        self._gpu = None
        gc.collect()
        if was_gpu:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # pragma: no cover - torch absent or no device
                logger.debug("document memory: could not empty the CUDA cache")


def _bounds(polygon) -> tuple[float, float, float, float] | None:
    """EasyOCR gives four corner points; reduce them to an axis-aligned box.

    Axis-aligned is enough for the one thing the box is used for -- deciding
    which regions share a line -- and is stable under the small rotations a
    warped page still carries.
    """
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, IndexError, ValueError):
        return None
    if not xs or not ys:
        return None
    return (min(xs), min(ys), max(xs), max(ys))
