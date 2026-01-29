"""Tests for PDF redaction application."""

import csv
import tempfile
from pathlib import Path

import pytest
import pymupdf

from portadoc.apply import (
    RedactionBox,
    load_redactions_from_csv,
    apply_redactions,
    apply_redactions_preview,
)


class TestLoadRedactionsFromCsv:
    """Test loading redaction boxes from CSV."""

    def test_load_redactions(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word_id", "page", "x0", "y0", "x1", "y1", "redact"])
            writer.writerow([0, 0, 10, 20, 50, 40, "true"])
            writer.writerow([1, 0, 60, 20, 100, 40, "false"])  # not marked
            writer.writerow([2, 1, 10, 20, 50, 40, "true"])
            path = Path(f.name)

        try:
            redactions = load_redactions_from_csv(path)

            assert len(redactions) == 2  # 2 pages with redactions
            assert 0 in redactions
            assert 1 in redactions

            # Page 0 should have 1 redaction (one was false)
            assert len(redactions[0]) == 1
            assert redactions[0][0] == RedactionBox(0, 10, 20, 50, 40)

            # Page 1 should have 1 redaction
            assert len(redactions[1]) == 1
            assert redactions[1][0] == RedactionBox(1, 10, 20, 50, 40)

        finally:
            path.unlink()

    def test_load_no_redactions(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word_id", "page", "x0", "y0", "x1", "y1", "redact"])
            writer.writerow([0, 0, 10, 20, 50, 40, "false"])
            path = Path(f.name)

        try:
            redactions = load_redactions_from_csv(path)
            assert len(redactions) == 0

        finally:
            path.unlink()


class TestApplyRedactions:
    """Test applying redactions to PDFs."""

    @pytest.fixture
    def sample_pdf(self):
        """Create a simple test PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            doc = pymupdf.open()
            page = doc.new_page(width=612, height=792)
            # Add some text
            page.insert_text((100, 100), "Peter Lou", fontsize=12)
            page.insert_text((100, 130), "7/24/25", fontsize=12)
            page.insert_text((100, 160), "MRN: 7829341", fontsize=12)
            doc.save(f.name)
            doc.close()
            yield Path(f.name)

        # Cleanup
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def sample_csv(self):
        """Create a CSV with redaction markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # CSV needs these columns: page, x0, y0, x1, y1, redact
            f.write("word_id,page,x0,y0,x1,y1,text,redact\n")
            # Coordinates roughly match where text would be
            f.write("0,0,95,85,160,105,Peter Lou,true\n")
            f.write("1,0,95,115,140,135,7/24/25,true\n")
            f.write("2,0,95,145,180,165,MRN: 7829341,false\n")  # MRN label not redacted
            f.flush()  # Flush before yielding
            path = Path(f.name)

        yield path
        path.unlink(missing_ok=True)

    def test_apply_redactions(self, sample_pdf, sample_csv):
        output_path = Path(tempfile.gettempdir()) / "redacted.pdf"

        try:
            count = apply_redactions(sample_pdf, sample_csv, output_path)
            assert count == 2  # Two redactions applied
            assert output_path.exists()

            # Verify PDF is valid
            doc = pymupdf.open(output_path)
            assert len(doc) == 1
            doc.close()

        finally:
            output_path.unlink(missing_ok=True)

    def test_apply_redactions_no_redactions(self, sample_pdf):
        # Create CSV with no redactions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["word_id", "page", "x0", "y0", "x1", "y1", "redact"])
            writer.writerow([0, 0, 100, 100, 200, 120, "false"])
            csv_path = Path(f.name)

        output_path = Path(tempfile.gettempdir()) / "no_redact.pdf"

        try:
            count = apply_redactions(sample_pdf, csv_path, output_path)
            assert count == 0
            assert output_path.exists()

        finally:
            csv_path.unlink()
            output_path.unlink(missing_ok=True)

    def test_apply_redactions_custom_color(self, sample_pdf, sample_csv):
        output_path = Path(tempfile.gettempdir()) / "redacted_white.pdf"

        try:
            count = apply_redactions(
                sample_pdf, sample_csv, output_path,
                color=(1, 1, 1)  # White
            )
            assert count == 2
            assert output_path.exists()

        finally:
            output_path.unlink(missing_ok=True)

    def test_apply_redactions_with_padding(self, sample_pdf, sample_csv):
        output_path = Path(tempfile.gettempdir()) / "redacted_padded.pdf"

        try:
            count = apply_redactions(
                sample_pdf, sample_csv, output_path,
                padding=5  # 5 point padding
            )
            assert count == 2
            assert output_path.exists()

        finally:
            output_path.unlink(missing_ok=True)


class TestApplyRedactionsPreview:
    """Test preview mode (outlines instead of filled boxes)."""

    @pytest.fixture
    def sample_pdf(self):
        """Create a simple test PDF."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            doc = pymupdf.open()
            page = doc.new_page(width=612, height=792)
            page.insert_text((100, 100), "Peter Lou", fontsize=12)
            doc.save(f.name)
            doc.close()
            yield Path(f.name)

        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def sample_csv(self):
        """Create a CSV with redaction markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("word_id,page,x0,y0,x1,y1,redact\n")
            f.write("0,0,95,85,160,105,true\n")
            f.flush()  # Flush before yielding
            path = Path(f.name)

        yield path
        path.unlink(missing_ok=True)

    def test_apply_preview(self, sample_pdf, sample_csv):
        output_path = Path(tempfile.gettempdir()) / "preview.pdf"

        try:
            count = apply_redactions_preview(sample_pdf, sample_csv, output_path)
            assert count == 1
            assert output_path.exists()

            # Verify PDF is valid
            doc = pymupdf.open(output_path)
            assert len(doc) == 1
            doc.close()

        finally:
            output_path.unlink(missing_ok=True)

    def test_apply_preview_custom_color(self, sample_pdf, sample_csv):
        output_path = Path(tempfile.gettempdir()) / "preview_blue.pdf"

        try:
            count = apply_redactions_preview(
                sample_pdf, sample_csv, output_path,
                border_color=(0, 0, 1)  # Blue
            )
            assert count == 1

        finally:
            output_path.unlink(missing_ok=True)
