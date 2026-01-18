"""Page orientation detection and correction."""

import numpy as np
import cv2
import pytesseract
from dataclasses import dataclass
from typing import Tuple, Optional
from PIL import Image


@dataclass
class OrientationResult:
    """Result of orientation detection."""
    angle: int           # Detected rotation: 0, 90, 180, or 270
    confidence: float    # Detection confidence (0-1)
    script: str          # Detected script (e.g., "Latin")
    method: str = "unknown"  # Detection method used


def detect_orientation_tesseract(image: np.ndarray) -> OrientationResult:
    """
    Detect page orientation using Tesseract OSD.

    Args:
        image: RGB image as numpy array

    Returns:
        OrientationResult with angle in {0, 90, 180, 270}
    """
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
        return OrientationResult(
            angle=osd.get("rotate", 0),
            confidence=osd.get("orientation_conf", 0) / 100.0,
            script=osd.get("script", "Unknown"),
            method="tesseract_osd"
        )
    except pytesseract.TesseractError:
        # OSD failed (e.g., not enough text)
        return OrientationResult(angle=0, confidence=0.0, script="Unknown", method="tesseract_osd")


# Lazy-loaded Surya predictors for orientation detection
_surya_det_predictor = None
_surya_rec_predictor = None


def _get_surya_predictors():
    """Get or create cached Surya predictors for orientation detection."""
    global _surya_det_predictor, _surya_rec_predictor

    if _surya_det_predictor is None:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.foundation import FoundationPredictor

        _surya_det_predictor = DetectionPredictor()
        foundation = FoundationPredictor()
        _surya_rec_predictor = RecognitionPredictor(foundation)

    return _surya_det_predictor, _surya_rec_predictor


def _get_surya_confidence(image: np.ndarray) -> Tuple[float, int]:
    """
    Get average OCR confidence for an image using Surya.

    Returns:
        Tuple of (average_confidence, word_count)
    """
    det_predictor, rec_predictor = _get_surya_predictors()
    pil_image = Image.fromarray(image)

    results = rec_predictor([pil_image], det_predictor=det_predictor, return_words=True)

    if not results or not results[0].text_lines:
        return 0.0, 0

    total_conf = 0.0
    count = 0
    for line in results[0].text_lines:
        for word in (line.words or [line]):
            if word.confidence:
                total_conf += word.confidence
                count += 1

    return (total_conf / count if count > 0 else 0.0), count


def detect_orientation_surya(
    image: np.ndarray,
    angles_to_test: list[int] | None = None
) -> OrientationResult:
    """
    Detect page orientation by running Surya OCR at multiple rotations.

    This method is slower but more accurate than Tesseract OSD for degraded images.
    It runs OCR at each test rotation and picks the one with highest average confidence.

    Args:
        image: RGB image as numpy array
        angles_to_test: Rotations to test (default: [0, 90, 180, 270])

    Returns:
        OrientationResult with the rotation angle that produces highest OCR confidence
    """
    if angles_to_test is None:
        angles_to_test = [0, 90, 180, 270]

    best_angle = 0
    best_confidence = 0.0
    best_word_count = 0

    for angle in angles_to_test:
        rotated = rotate_image(image, angle)
        avg_conf, word_count = _get_surya_confidence(rotated)

        # Prefer higher confidence, use word count as tiebreaker
        if avg_conf > best_confidence or (avg_conf == best_confidence and word_count > best_word_count):
            best_angle = angle
            best_confidence = avg_conf
            best_word_count = word_count

    return OrientationResult(
        angle=best_angle,
        confidence=best_confidence,
        script="Unknown",
        method="surya"
    )


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate image by specified angle (must be 0, 90, 180, or 270).

    Args:
        image: Image as numpy array
        angle: Rotation angle (0, 90, 180, or 270)

    Returns:
        Rotated image
    """
    if angle == 0:
        return image
    elif angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")


def transform_bbox_to_original(
    x0: float, y0: float, x1: float, y1: float,
    angle: int,
    rotated_width: float, rotated_height: float
) -> Tuple[float, float, float, float]:
    """
    Transform a bounding box from rotated coordinates back to original coordinates.

    After we rotate an image for OCR, the bboxes are in rotated space. This function
    transforms them back to the original (unrotated) coordinate space so they align
    with the original PDF when displayed.

    Args:
        x0, y0, x1, y1: Bbox in rotated image coordinates
        angle: The rotation that was applied (0, 90, 180, 270)
        rotated_width: Width of the rotated image
        rotated_height: Height of the rotated image

    Returns:
        Tuple of (x0, y0, x1, y1) in original image coordinates
    """
    if angle == 0:
        return x0, y0, x1, y1

    elif angle == 180:
        # 180° rotation: (x, y) -> (width - x, height - y)
        # Original dimensions are same as rotated
        new_x0 = rotated_width - x1
        new_y0 = rotated_height - y1
        new_x1 = rotated_width - x0
        new_y1 = rotated_height - y0
        return new_x0, new_y0, new_x1, new_y1

    elif angle == 90:
        # 90° CCW rotation: rotated is (orig_height, orig_width)
        # Point (x, y) on rotated -> (y, orig_width - x) on original
        # Original width = rotated_height, original height = rotated_width
        orig_width = rotated_height
        new_x0 = y0
        new_y0 = orig_width - x1
        new_x1 = y1
        new_y1 = orig_width - x0
        return new_x0, new_y0, new_x1, new_y1

    elif angle == 270:
        # 270° CW rotation: rotated is (orig_height, orig_width)
        # Point (x, y) on rotated -> (orig_height - y, x) on original
        # Original height = rotated_width
        orig_height = rotated_width
        new_x0 = orig_height - y1
        new_y0 = x0
        new_x1 = orig_height - y0
        new_y1 = x1
        return new_x0, new_y0, new_x1, new_y1

    else:
        raise ValueError(f"Unsupported rotation angle: {angle}")


def align_page(
    image: np.ndarray,
    method: str = "tesseract_osd",
    min_confidence: float = 0.1,
    allowed_angles: list[int] | None = None,
    surya_fallback_threshold: float = 0.05
) -> Tuple[np.ndarray, OrientationResult]:
    """
    Detect and correct page orientation.

    Args:
        image: RGB image as numpy array
        method: Detection method:
            - "tesseract_osd": Fast Tesseract OSD (default)
            - "surya": Slower but more accurate Surya-based detection
            - "auto": Use Tesseract OSD, fall back to Surya if confidence is very low
        min_confidence: Minimum confidence to apply rotation
        allowed_angles: Which rotations to correct (subset of [90, 180, 270])
        surya_fallback_threshold: If using "auto" method, use Surya when
            Tesseract OSD confidence is below this threshold

    Returns:
        Tuple of (corrected_image, orientation_result)
    """
    if allowed_angles is None:
        allowed_angles = [90, 180, 270]

    if method == "tesseract_osd":
        result = detect_orientation_tesseract(image)
    elif method == "surya":
        # Only test 0 and allowed angles
        angles_to_test = [0] + [a for a in allowed_angles if a != 0]
        result = detect_orientation_surya(image, angles_to_test)
    elif method == "auto":
        # Try Tesseract OSD first, fall back to Surya if confidence is very low
        result = detect_orientation_tesseract(image)
        if result.confidence < surya_fallback_threshold:
            # Tesseract OSD failed or very low confidence, try Surya
            angles_to_test = [0] + [a for a in allowed_angles if a != 0]
            result = detect_orientation_surya(image, angles_to_test)
    else:
        raise ValueError(f"Unknown orientation detection method: {method}")

    # Only rotate if confident and angle is in allowed list
    if result.confidence >= min_confidence and result.angle in allowed_angles:
        corrected = rotate_image(image, result.angle)
        return corrected, result

    return image, result
