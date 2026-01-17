"""Data models for Portadoc."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BBox:
    """Bounding box in PDF coordinate space (points, origin top-left)."""

    x0: float  # Left edge
    y0: float  # Top edge
    x1: float  # Right edge
    y1: float  # Bottom edge

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside this bbox."""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def iou(self, other: "BBox") -> float:
        """Calculate Intersection over Union with another bbox."""
        inter_x0 = max(self.x0, other.x0)
        inter_y0 = max(self.y0, other.y0)
        inter_x1 = min(self.x1, other.x1)
        inter_y1 = min(self.y1, other.y1)

        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            return 0.0

        inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
        union_area = self.area + other.area - inter_area

        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class Word:
    """A single word extracted from a document."""

    word_id: int
    text: str
    bbox: BBox
    page: int
    engine: str = ""  # Empty string means harmonized
    confidence: float = 0.0

    # Optional metadata
    tesseract_confidence: Optional[float] = None
    easyocr_confidence: Optional[float] = None


@dataclass
class HarmonizedWord:
    """
    Word with full tracking for smart harmonization output.

    Includes raw text from each engine and Levenshtein distances.
    """
    word_id: int
    page: int
    bbox: BBox
    text: str                    # Final voted text
    status: str                  # word|low_conf|pixel|secondary_only
    source: str                  # T|TE|TED|E|D|P|...
    confidence: float

    # Raw text from each engine (empty string if not detected)
    tess_text: str = ""
    easy_text: str = ""
    doctr_text: str = ""
    paddle_text: str = ""

    # Levenshtein distances to final text (-1 if engine didn't detect)
    dist_tess: int = -1
    dist_easy: int = -1
    dist_doctr: int = -1
    dist_paddle: int = -1

    def to_word(self) -> Word:
        """Convert to basic Word for backwards compatibility."""
        return Word(
            word_id=self.word_id,
            text=self.text,
            bbox=self.bbox,
            page=self.page,
            engine="",
            confidence=self.confidence,
        )


@dataclass
class Page:
    """A single page of extracted words."""

    page_number: int
    words: list[Word] = field(default_factory=list)
    width: float = 0.0  # Page width in points
    height: float = 0.0  # Page height in points


@dataclass
class Document:
    """A complete document with all extracted words."""

    filename: str
    pages: list[Page] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return sum(len(page.words) for page in self.pages)

    def all_words(self) -> list[Word]:
        """Return all words across all pages, flattened."""
        return [word for page in self.pages for word in page.words]
