"""Rendered documents with KNOWN text, at KNOWN poses.

SYNTHETIC, NOT PHYSICAL.

The point of this harness is that the answer is available independently of
the code under test. OCR accuracy is measured against the exact string
that was rendered; page detection is measured against the exact corners
the page was placed at; retrieval is measured against which document
actually contains a term.

A test that asserts OCR output against OCR output measures nothing.
"""

import io

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAGE_SIZE = (800, 1040)  # roughly A4 proportions

TRANSFORMER_PAPER = [
    "Attention Is All You Need",
    "",
    "Abstract",
    "The dominant sequence transduction models are",
    "based on complex recurrent or convolutional",
    "neural networks that include an encoder and a",
    "decoder. We propose a new simple network",
    "architecture, the Transformer, based solely on",
    "attention mechanisms, dispensing with recurrence",
    "and convolutions entirely.",
]

DEPTH_NOTES = [
    "Monocular Depth Notes",
    "",
    "Relative depth is not metric distance.",
    "A single prediction is evidence, not truth.",
    "Multi view consistency outranks one frame",
    "confidence. Scale must never be fabricated",
    "from a model that cannot measure it.",
]

RECEIPT = [
    "HARDWARE STORE",
    "",
    "Return policy: 30 days with receipt",
    "Item: cable ties        4.99",
    "Item: tape measure     12.50",
    "Total                  17.49",
]


# Font files to try, in order, for the rendered pages these fixtures use
# as OCR ground truth. The bare names resolve on Windows and on most
# Linux images; the absolute paths are macOS, which ships none of them.
#
# WHY THE ABSOLUTE PATHS MATTER, AND WHAT THEIR ABSENCE COST.
#
# `ImageFont.load_default()` with no size returns a 10-PIXEL BITMAP font
# and silently ignores the `font_size` these fixtures ask for. On macOS,
# where none of the three bare names resolves, every "printed page" this
# module rendered therefore came out very nearly blank -- measured, a
# full page of body text at a requested 34 px had a mean luminance of
# 249 of 255. Nothing downstream could see it: the capture gate found no
# text regions, `detect_page` returned None, no dwell ever opened, and
# roughly thirty Document Memory tests failed asserting that a page they
# believed they had drawn was recorded.
#
# None of that was a Tower defect. It is the fixture failing to draw,
# and it fails SILENTLY -- a missing font is not an error, it is a
# smaller font -- which is the reason it read as a product failure for a
# whole platform. The size-carrying `load_default(size=)` is the last
# resort rather than the first, because a real TrueType face is what the
# thresholds in `gate.py` were derived against.
_FONT_CANDIDATES = (
    "arial.ttf",
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _font(size: int):
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        # Pillow >= 10.1. Still a bitmap face, but at the size asked for
        # rather than at 10 px, so a page is at least legible ink.
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_page(
    lines: list[str], size=PAGE_SIZE, font_size: int = 34, margin: int = 60
) -> np.ndarray:
    """A printed page as a BGR image. The text is the ground truth."""
    image = Image.new("RGB", size, (252, 251, 248))
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    y = margin
    for line in lines:
        if line:
            draw.text((margin, y), line, fill=(18, 18, 20), font=font)
        y += int(font_size * 1.5)
    return np.array(image)[:, :, ::-1].copy()


def page_text(lines: list[str]) -> str:
    """Exactly what was rendered, as one string. The OCR ground truth.

    Space-joined, matching how a recogniser joins its regions -- so a
    comparison against OCR output compares like with like.
    """
    return " ".join(line for line in lines if line)


def page_regions(lines: list[str]) -> str:
    """The same text, newline-separated, for FixedTextRecogniser.

    Newlines mark REGION boundaries, which is the shape real OCR returns:
    one region per detected line. Feeding the fake an undivided blob
    would let tests pass on structure the real engine never provides.
    """
    return chr(10).join(line for line in lines if line)


def _default_corners(width: int, height: int, tilt: float) -> np.ndarray:
    """A page filling most of the frame, tilted by `tilt` (0 = square on)."""
    inset_x = width * 0.10
    inset_y = height * 0.06
    shift = width * 0.18 * tilt
    return np.array(
        [
            [inset_x + shift, inset_y],
            [width - inset_x, inset_y + height * 0.04 * tilt],
            [width - inset_x, height - inset_y - height * 0.04 * tilt],
            [inset_x + shift, height - inset_y],
        ],
        dtype=np.float32,
    )


def place_page(
    page: np.ndarray,
    frame_size=(640, 480),
    tilt: float = 0.0,
    corners: np.ndarray | None = None,
    background: int = 70,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite a page into a frame. Returns (frame_bgr, corners).

    The returned corners are the ground truth a detector must recover.
    The background is deliberately dark and textured so the page edge is
    findable but the scene is not a uniform void.
    """
    width, height = frame_size
    if corners is None:
        corners = _default_corners(width, height, tilt)

    rng = np.random.default_rng(11)
    frame = np.full((height, width, 3), background, dtype=np.uint8)
    noise = rng.integers(-12, 12, (height, width, 3), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    page_height, page_width = page.shape[:2]
    source = np.array(
        [
            [0, 0],
            [page_width - 1, 0],
            [page_width - 1, page_height - 1],
            [0, page_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    warped = cv2.warpPerspective(page, matrix, (width, height))
    mask = cv2.warpPerspective(
        np.full((page_height, page_width), 255, np.uint8), matrix, (width, height)
    )
    frame[mask > 0] = warped[mask > 0]
    return frame, corners.astype(np.float32)


def blank_page_frame(frame_size=(640, 480)) -> np.ndarray:
    """A page-shaped region with NO text. Must not be called a document."""
    blank = np.full((PAGE_SIZE[1], PAGE_SIZE[0], 3), 250, dtype=np.uint8)
    frame, _ = place_page(blank, frame_size=frame_size)
    return frame


def no_page_frame(frame_size=(640, 480)) -> np.ndarray:
    """A textured scene with nothing page-like in it."""
    width, height = frame_size
    rng = np.random.default_rng(5)
    small = rng.integers(40, 200, (height // 10, width // 10, 3), dtype=np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def encode(frame: np.ndarray, quality: int = 92) -> bytes:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    assert ok
    return buffer.tobytes()


def blur(frame: np.ndarray, kernel: int = 15) -> np.ndarray:
    return cv2.GaussianBlur(frame, (kernel, kernel), 0)


def document_frames(
    lines: list[str],
    count: int,
    frame_size=(640, 480),
    tilt: float = 0.0,
    jitter: float = 1.5,
) -> list[bytes]:
    """A held-still view of one page, with realistic small hand motion.

    `jitter` is in pixels. Zero would be a tripod, which is not what a
    person holding a page looks like, and a dwell tracker that only works
    on a perfectly static image would pass here and fail in a room.
    """
    page = render_page(lines)
    rng = np.random.default_rng(3)
    width, height = frame_size
    base = _default_corners(width, height, tilt)
    frames = []
    for _ in range(count):
        corners = base + rng.normal(0.0, jitter, base.shape).astype(np.float32)
        frame, _ = place_page(page, frame_size=frame_size, corners=corners)
        frames.append(encode(frame))
    return frames


def as_pil_jpeg(frame: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(frame[:, :, ::-1]).save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()
