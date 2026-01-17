"""Pixel-based detection for OCR-missed content."""

import cv2
import numpy as np

from .models import BBox, Word


def bbox_overlaps_any(
    bbox: BBox,
    existing_bboxes: list[BBox],
    overlap_threshold: float = 0.1,
) -> bool:
    """
    Check if a bounding box overlaps with any existing boxes.

    Uses a simple overlap check: if any corner of the new box is inside
    an existing box, or vice versa.
    """
    for existing in existing_bboxes:
        # Check IoU
        if bbox.iou(existing) > overlap_threshold:
            return True

        # Also check if centers are close (within half the size)
        new_cx = (bbox.x0 + bbox.x1) / 2
        new_cy = (bbox.y0 + bbox.y1) / 2
        ex_cx = (existing.x0 + existing.x1) / 2
        ex_cy = (existing.y0 + existing.y1) / 2

        # If the new box center is inside an existing box
        if (existing.x0 <= new_cx <= existing.x1 and
            existing.y0 <= new_cy <= existing.y1):
            return True

    return False


def detect_logo_regions(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    existing_bboxes: list[BBox] = None,
    min_area_pts: float = 1000,  # Minimum area in PDF points^2
    min_dimension_pts: float = 30,  # Minimum width/height in pts
) -> list[Word]:
    """
    Detect logo/image regions that OCR may have missed.

    Targets larger, boxy regions that are not text-like.
    """
    img_height, img_width = image.shape[:2]
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Binary threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Dilate to merge nearby pixels into regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(binary, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    existing_bboxes = existing_bboxes or []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Convert to PDF coordinates
        pdf_x0 = x * scale_x
        pdf_y0 = y * scale_y
        pdf_x1 = (x + w) * scale_x
        pdf_y1 = (y + h) * scale_y
        pdf_w = pdf_x1 - pdf_x0
        pdf_h = pdf_y1 - pdf_y0
        pdf_area = pdf_w * pdf_h

        # Filter: must be large enough
        if pdf_area < min_area_pts:
            continue

        # Filter: must be reasonably sized (not text-like)
        if pdf_w < min_dimension_pts and pdf_h < min_dimension_pts:
            continue

        # Filter: aspect ratio should be boxy (not extreme like lines)
        aspect = max(pdf_w, pdf_h) / max(min(pdf_w, pdf_h), 0.1)
        if aspect > 20:  # Skip line-like shapes
            continue

        new_bbox = BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1)

        if not bbox_overlaps_any(new_bbox, existing_bboxes):
            words.append(Word(
                word_id=-1,
                text="",
                bbox=new_bbox,
                page=page_num,
                engine="pixel_detector",
                confidence=0.0,
            ))

    return words


def detect_horizontal_lines(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    existing_bboxes: list[BBox] = None,
    min_width_pts: float = 100,  # Minimum line width in pts
    max_height_pts: float = 10,  # Maximum line height in pts
) -> list[Word]:
    """
    Detect horizontal lines (separators, underlines, etc).
    """
    img_height, img_width = image.shape[:2]
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    min_width_px = int(min_width_pts / scale_x)
    max_height_px = int(max_height_pts / scale_y)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Binary threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphological operation to detect horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_width_px // 4, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    existing_bboxes = existing_bboxes or []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        pdf_x0 = x * scale_x
        pdf_y0 = y * scale_y
        pdf_x1 = (x + w) * scale_x
        pdf_y1 = (y + h) * scale_y
        pdf_w = pdf_x1 - pdf_x0
        pdf_h = pdf_y1 - pdf_y0

        # Must be wide and thin
        if pdf_w < min_width_pts or pdf_h > max_height_pts:
            continue

        # Aspect ratio check (very horizontal)
        if pdf_w / max(pdf_h, 0.1) < 10:
            continue

        new_bbox = BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1)

        if not bbox_overlaps_any(new_bbox, existing_bboxes):
            words.append(Word(
                word_id=-1,
                text="",
                bbox=new_bbox,
                page=page_num,
                engine="pixel_detector",
                confidence=0.0,
            ))

    return words


def detect_vertical_lines(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    existing_bboxes: list[BBox] = None,
    min_height_pts: float = 50,  # Minimum line height in pts
    max_width_pts: float = 10,  # Maximum line width in pts
) -> list[Word]:
    """
    Detect vertical lines (margins, separators, etc).
    """
    img_height, img_width = image.shape[:2]
    scale_x = page_width / img_width
    scale_y = page_height / img_height

    min_height_px = int(min_height_pts / scale_y)

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Binary threshold
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Morphological operation to detect vertical lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_height_px // 4))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    words = []
    existing_bboxes = existing_bboxes or []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        pdf_x0 = x * scale_x
        pdf_y0 = y * scale_y
        pdf_x1 = (x + w) * scale_x
        pdf_y1 = (y + h) * scale_y
        pdf_w = pdf_x1 - pdf_x0
        pdf_h = pdf_y1 - pdf_y0

        # Must be tall and thin
        if pdf_h < min_height_pts or pdf_w > max_width_pts:
            continue

        # Aspect ratio check (very vertical)
        if pdf_h / max(pdf_w, 0.1) < 5:
            continue

        new_bbox = BBox(x0=pdf_x0, y0=pdf_y0, x1=pdf_x1, y1=pdf_y1)

        if not bbox_overlaps_any(new_bbox, existing_bboxes):
            words.append(Word(
                word_id=-1,
                text="",
                bbox=new_bbox,
                page=page_num,
                engine="pixel_detector",
                confidence=0.0,
            ))

    return words


def detect_missed_content(
    image: np.ndarray,
    page_num: int,
    page_width: float,
    page_height: float,
    existing_bboxes: list[BBox] = None,
) -> list[Word]:
    """
    Main function to detect content missed by OCR.

    Targets:
    1. Logo/image regions (large, boxy)
    2. Horizontal lines (separators, underlines)
    3. Vertical lines (margins, borders)
    """
    all_detected = []
    existing = list(existing_bboxes or [])

    # Detect horizontal lines first
    h_lines = detect_horizontal_lines(
        image, page_num, page_width, page_height,
        existing_bboxes=existing
    )
    all_detected.extend(h_lines)
    existing.extend([w.bbox for w in h_lines])

    # Detect vertical lines
    v_lines = detect_vertical_lines(
        image, page_num, page_width, page_height,
        existing_bboxes=existing
    )
    all_detected.extend(v_lines)
    existing.extend([w.bbox for w in v_lines])

    # Detect logo regions
    logos = detect_logo_regions(
        image, page_num, page_width, page_height,
        existing_bboxes=existing
    )
    all_detected.extend(logos)

    return all_detected
