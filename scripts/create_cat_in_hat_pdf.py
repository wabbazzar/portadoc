#!/usr/bin/env python3
"""Create a simple 'Cat in the Hat' test PDF for VLM fusion TDD."""

import fitz  # PyMuPDF


def create_cat_in_hat_pdf(output_path: str = "data/input/cat_in_hat.pdf"):
    """
    Create a test PDF with multiple lines at different font sizes.

    Line 1 (40pt): "The cat in the hat"
    Line 2 (40pt): "did a flip and a splat,"
    Line 3 (20pt): "then he tipped his tall hat and said, 'How about that!'"
    """
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)  # Letter size

    # Line 1: 40pt font - first part
    line1a = "The cat in the hat"
    point1a = fitz.Point(50, 80)
    page.insert_text(
        point1a,
        line1a,
        fontsize=40,
        fontname="helv",
    )

    # Line 2: 40pt font - second part (continuation)
    line1b = "did a flip and a splat,"
    point1b = fitz.Point(50, 130)  # Below line 1a
    page.insert_text(
        point1b,
        line1b,
        fontsize=40,
        fontname="helv",
    )

    # Line 3: 20pt font
    line2 = "then he tipped his tall hat and said, 'How about that!'"
    point2 = fitz.Point(50, 200)
    page.insert_text(
        point2,
        line2,
        fontsize=20,
        fontname="helv",
    )

    doc.save(output_path)
    doc.close()
    print(f"Created: {output_path}")
    print(f"Line 1a (40pt): {line1a}")
    print(f"Line 1b (40pt): {line1b}")
    print(f"Line 2 (20pt): {line2}")


if __name__ == "__main__":
    create_cat_in_hat_pdf()
