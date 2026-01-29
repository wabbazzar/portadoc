"""Entity detection and redaction marking for extracted words."""

import csv
import re
from enum import Enum
from pathlib import Path
from typing import TextIO

from .patterns import matches_date, matches_code, matches_exclusion


class EntityType(Enum):
    """Type of detected entity."""
    NONE = ""
    NAME = "NAME"
    DATE = "DATE"
    CODE = "CODE"


# Default path to names dictionary
DEFAULT_NAMES_PATH = Path(__file__).parent.parent.parent / "data" / "dictionaries" / "us_names.txt"


def load_names(path: Path | None = None) -> set[str]:
    """
    Load names from dictionary file into a set for O(1) lookup.

    Args:
        path: Path to names file (one name per line).
              Defaults to data/dictionaries/us_names.txt.

    Returns:
        Set of lowercase names.
    """
    if path is None:
        path = DEFAULT_NAMES_PATH

    if not path.exists():
        return set()

    names = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name and not name.startswith("#"):
                names.add(name.lower())
    return names


def strip_punctuation(text: str) -> str:
    """Remove leading/trailing punctuation for matching."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)


class EntityDetector:
    """
    Fast entity detector using regex and dictionary lookup.

    Detection order (most specific first):
    1. Check exclusions (time, currency, percent) -> skip
    2. Check date patterns -> DATE
    3. Check code patterns -> CODE
    4. Check names dictionary -> NAME
    """

    def __init__(
        self,
        names_path: Path | None = None,
        detect_names: bool = True,
        detect_dates: bool = True,
        detect_codes: bool = True,
    ):
        """
        Initialize the entity detector.

        Args:
            names_path: Path to names dictionary file.
            detect_names: Whether to detect name entities.
            detect_dates: Whether to detect date entities.
            detect_codes: Whether to detect code entities.
        """
        self._names: set[str] = set()
        self._detect_names = detect_names
        self._detect_dates = detect_dates
        self._detect_codes = detect_codes

        if detect_names:
            self._names = load_names(names_path)

    def detect(self, text: str) -> tuple[EntityType, bool]:
        """
        Detect entity type for a word.

        Args:
            text: The word text to analyze.

        Returns:
            Tuple of (entity, should_redact).
            - entity: EntityType.NONE if not an entity
            - should_redact: True if the word should be redacted
        """
        if not text or not text.strip():
            return EntityType.NONE, False

        text = text.strip()

        # Check exclusions first (false positives)
        if matches_exclusion(text):
            return EntityType.NONE, False

        # Check dates
        if self._detect_dates and matches_date(text):
            return EntityType.DATE, True

        # Check codes
        if self._detect_codes and matches_code(text):
            return EntityType.CODE, True

        # Check names (case-insensitive, strip punctuation)
        if self._detect_names:
            clean_text = strip_punctuation(text)
            if clean_text and clean_text.lower() in self._names:
                return EntityType.NAME, True

        return EntityType.NONE, False

    def detect_batch(self, texts: list[str]) -> list[tuple[EntityType, bool]]:
        """Detect entities for a batch of texts."""
        return [self.detect(text) for text in texts]


def redact_csv(
    input_csv: Path,
    output_csv: Path | None = None,
    names_path: Path | None = None,
    detect_names: bool = True,
    detect_dates: bool = True,
    detect_codes: bool = True,
) -> int:
    """
    Read a CSV, detect entities, and append entity and redact columns.

    Args:
        input_csv: Path to input CSV (from portadoc extract).
        output_csv: Path to output CSV. If None, overwrites input.
        names_path: Path to custom names dictionary.
        detect_names: Whether to detect name entities.
        detect_dates: Whether to detect date entities.
        detect_codes: Whether to detect code entities.

    Returns:
        Number of words marked for redaction.
    """
    detector = EntityDetector(
        names_path=names_path,
        detect_names=detect_names,
        detect_dates=detect_dates,
        detect_codes=detect_codes,
    )

    # Read input CSV
    rows = []
    fieldnames = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    # Add new columns if not present
    if "entity" not in fieldnames:
        fieldnames.append("entity")
    if "redact" not in fieldnames:
        fieldnames.append("redact")

    # Detect entities and mark redactions
    redact_count = 0
    for row in rows:
        text = row.get("text", "")
        entity, should_redact = detector.detect(text)

        row["entity"] = entity.value
        row["redact"] = "true" if should_redact else "false"

        if should_redact:
            redact_count += 1

    # Write output CSV
    out_path = output_csv if output_csv else input_csv
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return redact_count


def get_redaction_stats(input_csv: Path) -> dict:
    """
    Get statistics about detected entities in a CSV.

    Args:
        input_csv: Path to CSV with entity and redact columns.

    Returns:
        Dictionary with counts by entity type.
    """
    stats = {
        "total_words": 0,
        "redacted_count": 0,
        "by_type": {
            "NAME": 0,
            "DATE": 0,
            "CODE": 0,
        },
    }

    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total_words"] += 1
            if row.get("redact") == "true":
                stats["redacted_count"] += 1
                entity = row.get("entity", "")
                if entity in stats["by_type"]:
                    stats["by_type"][entity] += 1

    return stats
