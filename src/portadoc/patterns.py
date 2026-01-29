"""Compiled regex patterns for entity detection."""

import re

# DATE patterns
DATE_PATTERNS = [
    # US date: 7/24/25, 12-31-2024, with optional trailing punctuation
    re.compile(r"^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}[,.]?$"),
    # ISO date: 2024-12-31
    re.compile(r"^\d{4}-\d{2}-\d{2}$"),
]

# CODE patterns (numeric identifiers)
CODE_PATTERNS = [
    # Phone: (555) 123-4567 or 555-123-4567 or 555.123.4567
    re.compile(r"^\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}$"),
    # SSN: 123-45-6789
    re.compile(r"^\d{3}-\d{2}-\d{4}$"),
    # Generic 5+ digit codes
    re.compile(r"^\d{5,}$"),
    # Alphanumeric codes: ABC12345, A1234567
    re.compile(r"^[A-Z]{1,3}\d{4,}$"),
    # MRN-style: 7829341 (7 digits)
    re.compile(r"^\d{7}$"),
]

# Exclusion patterns (false positives to avoid)
EXCLUSION_PATTERNS = [
    # Time: 10:28, 2:30
    re.compile(r"^\d{1,2}:\d{2}$"),
    # Currency: $100, $1,000.00
    re.compile(r"^\$[\d,]+\.?\d*$"),
    # Percent: 50%, 3.5%
    re.compile(r"^\d+\.?\d*%$"),
    # Common abbreviations that look like codes
    re.compile(r"^(AM|PM|ID|MRN|DOB|SSN|EIN)$", re.IGNORECASE),
]


def matches_date(text: str) -> bool:
    """Check if text matches a date pattern."""
    return any(p.match(text) for p in DATE_PATTERNS)


def matches_code(text: str) -> bool:
    """Check if text matches a code pattern."""
    return any(p.match(text) for p in CODE_PATTERNS)


def matches_exclusion(text: str) -> bool:
    """Check if text matches an exclusion pattern (false positive)."""
    return any(p.match(text) for p in EXCLUSION_PATTERNS)
