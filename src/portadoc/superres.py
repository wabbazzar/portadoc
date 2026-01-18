"""Image super-resolution for improving OCR on degraded documents."""

import os
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np

# Model paths - relative to package or absolute
# __file__ is src/portadoc/superres.py, so parent.parent.parent = portadoc/
_MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# Cached super-resolution instances
# Note: Type annotation uses Any because DnnSuperResImpl may not be available in all cv2 builds
_sr_cache: dict[tuple[str, int], "cv2.dnn_superres.DnnSuperResImpl"] = {}


def _get_model_path(model_name: str, scale: int) -> Path:
    """Get the path to a super-resolution model file."""
    filename = f"{model_name.upper()}_x{scale}.pb"
    model_path = _MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Super-resolution model not found: {model_path}\n"
            f"Download from: https://github.com/fannymonori/TF-ESPCN"
        )

    return model_path


def _get_sr_instance(method: str, scale: int) -> "cv2.dnn_superres.DnnSuperResImpl":
    """Get or create a cached super-resolution instance."""
    cache_key = (method, scale)

    if cache_key not in _sr_cache:
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_path = _get_model_path(method, scale)
        sr.readModel(str(model_path))
        sr.setModel(method.lower(), scale)
        _sr_cache[cache_key] = sr

    return _sr_cache[cache_key]


def upscale_image(
    image: np.ndarray,
    scale: int = 4,
    method: Literal["espcn", "fsrcnn", "bicubic", "lanczos"] = "espcn",
) -> np.ndarray:
    """
    Upscale an image using super-resolution.

    Args:
        image: Input image as numpy array (BGR or RGB, uint8)
        scale: Upscale factor (2 or 4 for DNN methods, any for interpolation)
        method: Super-resolution method:
            - "espcn": Fast DNN-based (recommended for speed)
            - "fsrcnn": Fast DNN-based (alternative)
            - "bicubic": OpenCV bicubic interpolation (fallback)
            - "lanczos": OpenCV Lanczos interpolation (fallback)

    Returns:
        Upscaled image as numpy array (same type as input)
    """
    if scale <= 1:
        return image

    # Use interpolation methods for non-standard scales or as fallback
    if method == "bicubic":
        return cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    elif method == "lanczos":
        return cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
        )

    # DNN-based super-resolution (only supports 2x and 4x)
    if scale not in (2, 4):
        # For non-standard scales, combine 2x/4x with interpolation
        if scale == 3:
            # 2x then 1.5x bicubic
            upscaled = upscale_image(image, 2, method)
            return cv2.resize(upscaled, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        elif scale > 4:
            # 4x then remaining bicubic
            upscaled = upscale_image(image, 4, method)
            remaining = scale / 4
            return cv2.resize(upscaled, None, fx=remaining, fy=remaining, interpolation=cv2.INTER_CUBIC)
        else:
            # Fallback to bicubic
            return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    try:
        sr = _get_sr_instance(method, scale)
        return sr.upsample(image)
    except Exception as e:
        # Fallback to bicubic if DNN fails
        import warnings
        warnings.warn(f"Super-resolution failed ({e}), falling back to bicubic")
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def is_superres_available(method: str = "espcn") -> bool:
    """Check if super-resolution models are available."""
    try:
        for scale in (2, 4):
            model_path = _get_model_path(method, scale)
            if not model_path.exists():
                return False
        return True
    except FileNotFoundError:
        return False


def get_available_methods() -> list[str]:
    """Get list of available super-resolution methods."""
    methods = ["bicubic", "lanczos"]  # Always available

    for dnn_method in ["espcn", "fsrcnn"]:
        if is_superres_available(dnn_method):
            methods.append(dnn_method)

    return methods
