"""Reading order reconstruction module.

Takes extracted words (already sorted in reading order) and reconstructs them into
readable lines and paragraphs for display.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LineWord:
    """A word within a reconstructed line."""
    word_id: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    status: str
    confidence: float


@dataclass
class ReconstructedLine:
    """A line of text reconstructed from words."""
    line_id: int
    text: str  # Joined text of all words
    words: list[LineWord] = field(default_factory=list)
    x0: float = 0.0  # Bounding box of entire line
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    @property
    def word_ids(self) -> list[int]:
        """Return list of word IDs in this line."""
        return [w.word_id for w in self.words]


def reconstruct_lines(
    words: list[dict],
    page: int,
    line_height_factor: float = 1.5,
) -> list[ReconstructedLine]:
    """
    Reconstruct lines from a list of words for a specific page.

    Words are expected to already be in reading order (sorted by word_id).
    This function groups consecutive words into lines based on:
    - Y-coordinate proximity (same visual line)
    - X-position continuity (not jumping back to start a new line)

    For multi-column documents, the reading order may jump between columns,
    so we detect new lines when:
    1. Y changes by more than line_height_factor * avg_word_height
    2. X position jumps backward significantly (new column/line start)

    Args:
        words: List of word dicts with keys: word_id, page, x0, y0, x1, y1, text, status, confidence
        page: Page number (0-indexed) to filter words
        line_height_factor: Multiplier for word height to determine line break threshold

    Returns:
        List of ReconstructedLine objects in reading order
    """
    # Filter to current page and sort by word_id (reading order)
    page_words = [w for w in words if w.get("page") == page]
    page_words.sort(key=lambda w: w.get("word_id", 0))

    if not page_words:
        return []

    # Calculate average word height for line break detection
    heights = [w.get("y1", 0) - w.get("y0", 0) for w in page_words if w.get("y1", 0) > w.get("y0", 0)]
    avg_height = sum(heights) / len(heights) if heights else 10.0
    line_threshold = avg_height * line_height_factor

    lines: list[ReconstructedLine] = []
    current_line: Optional[ReconstructedLine] = None
    line_id = 0

    for word in page_words:
        word_obj = LineWord(
            word_id=word.get("word_id", 0),
            text=word.get("text", ""),
            x0=word.get("x0", 0),
            y0=word.get("y0", 0),
            x1=word.get("x1", 0),
            y1=word.get("y1", 0),
            status=word.get("status", ""),
            confidence=word.get("confidence", 0),
        )

        if current_line is None:
            # Start first line
            current_line = ReconstructedLine(
                line_id=line_id,
                text=word_obj.text,
                words=[word_obj],
                x0=word_obj.x0,
                y0=word_obj.y0,
                x1=word_obj.x1,
                y1=word_obj.y1,
            )
        else:
            # Calculate line metrics
            last_word = current_line.words[-1]
            line_center_y = (current_line.y0 + current_line.y1) / 2
            word_center_y = (word_obj.y0 + word_obj.y1) / 2
            y_diff = abs(word_center_y - line_center_y)

            # Check for new line conditions:
            # 1. Y jumps more than line_threshold (different visual line)
            # 2. X jumps backward significantly (new line/column start)
            x_jump_back = word_obj.x0 < last_word.x0 - avg_height * 2

            is_same_line = y_diff <= line_threshold and not x_jump_back

            if is_same_line:
                # Same line - append word
                current_line.text += " " + word_obj.text
                current_line.words.append(word_obj)
                # Update line bbox
                current_line.x1 = max(current_line.x1, word_obj.x1)
                current_line.y0 = min(current_line.y0, word_obj.y0)
                current_line.y1 = max(current_line.y1, word_obj.y1)
            else:
                # New line
                lines.append(current_line)
                line_id += 1
                current_line = ReconstructedLine(
                    line_id=line_id,
                    text=word_obj.text,
                    words=[word_obj],
                    x0=word_obj.x0,
                    y0=word_obj.y0,
                    x1=word_obj.x1,
                    y1=word_obj.y1,
                )

    # Don't forget the last line
    if current_line is not None:
        lines.append(current_line)

    return lines


def lines_to_json(lines: list[ReconstructedLine]) -> list[dict]:
    """Convert ReconstructedLine objects to JSON-serializable dicts."""
    return [
        {
            "line_id": line.line_id,
            "text": line.text,
            "word_ids": line.word_ids,
            "words": [
                {
                    "word_id": w.word_id,
                    "text": w.text,
                    "x0": w.x0,
                    "y0": w.y0,
                    "x1": w.x1,
                    "y1": w.y1,
                    "status": w.status,
                    "confidence": w.confidence,
                }
                for w in line.words
            ],
            "bbox": {
                "x0": line.x0,
                "y0": line.y0,
                "x1": line.x1,
                "y1": line.y1,
            },
        }
        for line in lines
    ]
