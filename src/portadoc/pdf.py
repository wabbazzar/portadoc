"""PDF loading and image conversion using PyMuPDF."""

from pathlib import Path
from typing import Iterator

import numpy as np
import pymupdf

from .models import Page


class PDFDocument:
    """Handles PDF loading and page-to-image conversion."""

    def __init__(self, path: str | Path, dpi: int = 300):
        """
        Initialize PDF document.

        Args:
            path: Path to PDF file
            dpi: Resolution for rendering pages to images (default 300)
        """
        self.path = Path(path)
        self.dpi = dpi
        self._doc = pymupdf.open(str(self.path))
        self._zoom = dpi / 72.0  # PDF standard is 72 DPI

    def __len__(self) -> int:
        return len(self._doc)

    def __enter__(self) -> "PDFDocument":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the PDF document."""
        if self._doc:
            self._doc.close()

    def get_page_size(self, page_num: int) -> tuple[float, float]:
        """
        Get page dimensions in PDF points.

        Returns:
            Tuple of (width, height) in points
        """
        page = self._doc[page_num]
        rect = page.rect
        return rect.width, rect.height

    def page_to_image(self, page_num: int) -> np.ndarray:
        """
        Render a PDF page to a numpy array (RGB image).

        Args:
            page_num: Page number (0-indexed)

        Returns:
            numpy array of shape (height, width, 3) with RGB values
        """
        page = self._doc[page_num]
        mat = pymupdf.Matrix(self._zoom, self._zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=pymupdf.csRGB)

        # Convert to numpy array
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )
        return img.copy()  # Return a copy to avoid memory issues

    def pages(self) -> Iterator[tuple[int, np.ndarray, float, float]]:
        """
        Iterate over all pages, yielding page images.

        Yields:
            Tuple of (page_number, image_array, page_width_pts, page_height_pts)
        """
        for i in range(len(self)):
            width, height = self.get_page_size(i)
            img = self.page_to_image(i)
            yield i, img, width, height

    def image_to_pdf_coords(
        self, img_x: float, img_y: float, img_width: int, img_height: int,
        page_width: float, page_height: float
    ) -> tuple[float, float]:
        """
        Convert image coordinates to PDF coordinates.

        Image coordinates: origin top-left, pixels
        PDF coordinates: origin top-left, points

        Args:
            img_x, img_y: Coordinates in image space (pixels)
            img_width, img_height: Image dimensions in pixels
            page_width, page_height: Page dimensions in PDF points

        Returns:
            Tuple of (pdf_x, pdf_y) in points
        """
        scale_x = page_width / img_width
        scale_y = page_height / img_height

        pdf_x = img_x * scale_x
        pdf_y = img_y * scale_y

        return pdf_x, pdf_y

    def image_bbox_to_pdf_bbox(
        self, img_bbox: tuple[float, float, float, float],
        img_width: int, img_height: int,
        page_width: float, page_height: float
    ) -> tuple[float, float, float, float]:
        """
        Convert image bounding box to PDF bounding box.

        Args:
            img_bbox: (x0, y0, x1, y1) in image pixels
            img_width, img_height: Image dimensions in pixels
            page_width, page_height: Page dimensions in PDF points

        Returns:
            (x0, y0, x1, y1) in PDF points
        """
        x0, y0, x1, y1 = img_bbox

        pdf_x0, pdf_y0 = self.image_to_pdf_coords(
            x0, y0, img_width, img_height, page_width, page_height
        )
        pdf_x1, pdf_y1 = self.image_to_pdf_coords(
            x1, y1, img_width, img_height, page_width, page_height
        )

        return pdf_x0, pdf_y0, pdf_x1, pdf_y1


def load_pdf(path: str | Path, dpi: int = 300) -> PDFDocument:
    """
    Load a PDF document.

    Args:
        path: Path to PDF file
        dpi: Resolution for rendering (default 300)

    Returns:
        PDFDocument instance
    """
    return PDFDocument(path, dpi)
