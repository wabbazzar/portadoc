"""Portadoc - PDF Word Extraction for Document Redaction."""

__version__ = "0.1.0"

from .models import BBox, Document, Page, Word, HarmonizedWord
from .extractor import extract_words, extract_document, extract_to_csv
from .pdf import load_pdf

__all__ = [
    "BBox",
    "Document",
    "Page",
    "Word",
    "HarmonizedWord",
    "extract_words",
    "extract_document",
    "extract_to_csv",
    "load_pdf",
]
