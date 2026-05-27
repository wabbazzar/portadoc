"""Regression tests for per-engine OCR failure resilience.

A single OCR engine crashing at runtime (e.g. a version-skew bug like Surya's
``pad_token_id`` failure) must NOT abort the whole extract. The pipeline should
log a warning and harmonize over the engines that succeeded.
"""

from pathlib import Path

import pytest

from portadoc import extractor
from portadoc.models import BBox, Word

PDF = Path(__file__).parent.parent / "data" / "input" / "peter_lou.pdf"


def _canned_tesseract(image, page_num, page_width, page_height, *args, **kwargs):
    """Fast, deterministic stand-in for a real Tesseract run."""
    return [
        Word(word_id=-1, text="Hello", bbox=BBox(10, 10, 50, 30),
             page=page_num, engine="tesseract", confidence=95.0),
        Word(word_id=-1, text="World", bbox=BBox(60, 10, 100, 30),
             page=page_num, engine="tesseract", confidence=95.0),
    ]


def _canned_surya(image, page_num, page_width, page_height, *args, **kwargs):
    """Fast, deterministic stand-in for a real Surya run."""
    return [
        Word(word_id=-1, text="Hello", bbox=BBox(11, 11, 51, 31),
             page=page_num, engine="surya", confidence=98.0),
    ]


def _boom(*args, **kwargs):
    raise RuntimeError("'SuryaDecoderConfig' object has no attribute 'pad_token_id'")


@pytest.fixture
def only_tess_and_surya(monkeypatch):
    """Enable only Tesseract (primary) + Surya (secondary), both stubbed."""
    monkeypatch.setattr(extractor, "is_tesseract_available", lambda: True)
    monkeypatch.setattr(extractor, "is_surya_available", lambda: True)
    monkeypatch.setattr(extractor, "is_easyocr_available", lambda: False)
    monkeypatch.setattr(extractor, "is_paddleocr_available", lambda: False)
    monkeypatch.setattr(extractor, "is_doctr_available", lambda: False)
    monkeypatch.setattr(extractor, "is_kraken_available", lambda: False)


def _run_extract(**overrides):
    kwargs = dict(
        pdf_path=PDF,
        dpi=72,
        use_easyocr=False,
        use_paddleocr=False,
        use_doctr=False,
        use_surya=True,
        use_kraken=False,
        use_pixel_detection=False,
        preprocess="none",
        align_pages=False,
    )
    kwargs.update(overrides)
    return extractor.extract_words(**kwargs)


def test_secondary_engine_crash_does_not_abort(only_tess_and_surya, monkeypatch):
    """Surya (secondary) raising must not abort; Tesseract words survive."""
    monkeypatch.setattr(extractor, "extract_words_tesseract", _canned_tesseract)
    monkeypatch.setattr(extractor, "extract_words_surya", _boom)

    words = _run_extract()

    assert words, "extract returned no words despite working primary engine"
    texts = {w.text for w in words}
    assert "Hello" in texts and "World" in texts


def test_primary_engine_crash_falls_back_to_survivor(only_tess_and_surya, monkeypatch):
    """If the configured primary crashes, a surviving engine becomes primary."""
    # Make Surya the primary engine, then have it crash.
    monkeypatch.setattr(extractor, "extract_words_surya", _boom)
    monkeypatch.setattr(extractor, "extract_words_tesseract", _canned_tesseract)

    words = _run_extract(primary_engine="surya")

    assert words, "extract aborted when primary engine crashed"
    texts = {w.text for w in words}
    assert "Hello" in texts and "World" in texts


def test_all_engines_crash_returns_empty_not_exception(only_tess_and_surya, monkeypatch):
    """Every engine failing yields empty output, not an unhandled exception."""
    monkeypatch.setattr(extractor, "extract_words_tesseract", _boom)
    monkeypatch.setattr(extractor, "extract_words_surya", _boom)

    words = _run_extract(use_pixel_detection=False)

    # No crash; pixel detection disabled, so no words.
    assert words == []


def test_probe_engine_reports_runtime_failure(monkeypatch):
    """probe_engine() reports FAILS (not OK) when an engine crashes at runtime."""
    monkeypatch.setattr(extractor, "is_surya_available", lambda: True)
    monkeypatch.setattr(extractor, "get_surya_version", lambda: "0.17.0")
    monkeypatch.setattr(extractor, "extract_words_surya", _boom)

    result = extractor.probe_engine("surya")

    assert result["status"] == "FAILS"
    assert "pad_token_id" in result["detail"]
