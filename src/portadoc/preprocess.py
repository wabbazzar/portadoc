"""OpenCV preprocessing pipeline for improved OCR accuracy."""

from enum import Enum
from typing import Optional

import cv2
import numpy as np


class PreprocessLevel(Enum):
    """Preprocessing intensity levels."""
    NONE = "none"
    LIGHT = "light"      # For clean PDFs
    STANDARD = "standard"  # Default for most documents
    AGGRESSIVE = "aggressive"  # For degraded/noisy documents
    DEGRADED = "degraded"  # For blurry/low-DPI documents: upscale + bilateral + CLAHE + sharpen


class UpscaleMethod(Enum):
    """Image upscaling interpolation methods."""
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    LANCZOS = "lanczos"  # Best for OCR - preserves edges


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert RGB image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def denoise(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply non-local means denoising.

    Args:
        image: Grayscale image
        strength: Denoising strength (higher = more smoothing)

    Returns:
        Denoised grayscale image
    """
    return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)


def enhance_contrast(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Args:
        image: Grayscale image
        clip_limit: Threshold for contrast limiting

    Returns:
        Contrast-enhanced grayscale image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image)


def binarize(
    image: np.ndarray,
    method: str = "otsu",
    block_size: int = 11,
    c: int = 2
) -> np.ndarray:
    """
    Convert grayscale to binary (black and white).

    Args:
        image: Grayscale image
        method: "otsu" for global threshold, "adaptive" for local threshold
        block_size: Block size for adaptive method (must be odd)
        c: Constant subtracted from mean for adaptive method

    Returns:
        Binary image (0 or 255 values)
    """
    if method == "otsu":
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, c
        )
    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Sharpen image using unsharp masking.

    Args:
        image: Input image
        strength: Sharpening strength (0-2 typical)

    Returns:
        Sharpened image
    """
    blurred = cv2.GaussianBlur(image, (0, 0), 3)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened


def deskew(image: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """
    Correct slight rotation/skew in document images.

    Args:
        image: Grayscale or binary image
        max_angle: Maximum angle to correct (degrees)

    Returns:
        Deskewed image
    """
    # Find edges
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 100:
        return image

    # Get rotation angle from minimum area rectangle
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]

    # Normalize angle
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct small angles
    if abs(angle) > max_angle:
        return image

    # Rotate image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def remove_borders(image: np.ndarray, margin: int = 5) -> np.ndarray:
    """
    Remove black borders from scanned documents.

    Args:
        image: Grayscale image
        margin: Pixels to keep from detected content

    Returns:
        Cropped image with borders removed
    """
    # Find non-white regions
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image

    x, y, w, h = cv2.boundingRect(coords)

    # Add margin
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)

    return image[y:y+h, x:x+w]


def preprocess_for_ocr(
    image: np.ndarray,
    level: PreprocessLevel = PreprocessLevel.STANDARD,
    return_rgb: bool = True
) -> np.ndarray:
    """
    Apply preprocessing pipeline for OCR.

    Args:
        image: RGB image as numpy array
        level: Preprocessing intensity level
        return_rgb: If True, convert result back to RGB for OCR engines

    Returns:
        Preprocessed image
    """
    if level == PreprocessLevel.NONE:
        return image

    # Convert to grayscale
    gray = to_grayscale(image)

    if level == PreprocessLevel.LIGHT:
        # Light preprocessing: just grayscale + slight sharpening
        result = sharpen(gray, strength=0.5)

    elif level == PreprocessLevel.STANDARD:
        # Standard: denoise + contrast + sharpening
        denoised = denoise(gray, strength=7)
        enhanced = enhance_contrast(denoised, clip_limit=1.5)
        result = sharpen(enhanced, strength=0.5)

    elif level == PreprocessLevel.AGGRESSIVE:
        # Aggressive: full pipeline for degraded documents
        denoised = denoise(gray, strength=12)
        enhanced = enhance_contrast(denoised, clip_limit=2.5)
        sharpened = sharpen(enhanced, strength=1.0)
        # Use adaptive binarization for degraded docs
        result = binarize(sharpened, method="adaptive", block_size=15, c=4)

    elif level == PreprocessLevel.DEGRADED:
        # Degraded: for blurry/low-DPI documents
        # Step 1: Upscale 2x with Lanczos interpolation to increase effective DPI
        h, w = gray.shape[:2]
        upscaled = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Bilateral filter preserves edges while reducing noise
        bilateral = cv2.bilateralFilter(upscaled, d=9, sigmaColor=75, sigmaSpace=75)

        # Step 3: CLAHE for contrast enhancement
        enhanced = enhance_contrast(bilateral, clip_limit=2.0)

        # Step 4: Unsharp mask for edge enhancement
        result = sharpen(enhanced, strength=1.5)

    else:
        result = gray

    # Convert back to RGB if requested (some OCR engines expect RGB)
    if return_rgb and len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    return result


def auto_detect_quality(image: np.ndarray) -> PreprocessLevel:
    """
    Automatically detect image quality and suggest preprocessing level.

    Args:
        image: RGB image

    Returns:
        Recommended preprocessing level
    """
    gray = to_grayscale(image)

    # Calculate noise level using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Calculate contrast using standard deviation
    contrast = np.std(gray)

    # High Laplacian variance = sharp image, low = blurry
    # High contrast std = good contrast

    if laplacian_var > 500 and contrast > 50:
        # Clean, sharp image
        return PreprocessLevel.LIGHT
    elif laplacian_var > 100 and contrast > 30:
        # Moderate quality
        return PreprocessLevel.STANDARD
    elif laplacian_var > 50:
        # Degraded/noisy image
        return PreprocessLevel.AGGRESSIVE
    else:
        # Very low quality/blurry image (low DPI, heavy compression)
        # laplacian_var < 50 indicates severe blur
        return PreprocessLevel.DEGRADED
