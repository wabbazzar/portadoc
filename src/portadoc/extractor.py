"""Main word extraction pipeline."""

from pathlib import Path
from typing import Optional

from .models import BBox, Document, Page, Word, HarmonizedWord
from .pdf import load_pdf
from .ocr.tesseract import extract_words_tesseract, is_tesseract_available
from .ocr.easyocr import extract_words_easyocr, is_easyocr_available
from .ocr.paddleocr import extract_words_paddleocr, is_paddleocr_available
from .ocr.doctr_ocr import extract_words_doctr, is_doctr_available
from .ocr.surya_ocr import extract_words_surya, is_surya_available
from .detection import detect_missed_content
from .harmonize import harmonize
from .preprocess import PreprocessLevel, preprocess_for_ocr, auto_detect_quality
from .superres import upscale_image
from .triage import TriageLevel, triage_words
from .config import PortadocConfig, get_config, load_config
from .geometric_clustering import order_words_by_reading
from .page_align import align_page, transform_bbox_to_original


def extract_words(
    pdf_path: str | Path,
    dpi: int = 300,
    use_tesseract: bool = True,
    use_easyocr: bool = True,
    use_paddleocr: bool = False,
    use_doctr: bool = False,
    use_surya: bool = True,
    use_pixel_detection: bool = True,
    gpu: bool = False,
    preprocess: Optional[str] = "auto",
    upscale: Optional[int] = None,
    upscale_method: str = "espcn",
    tesseract_psm: int = 3,
    tesseract_oem: int = 3,
    easyocr_decoder: str = "greedy",
    easyocr_text_threshold: float = 0.7,
    config_path: Optional[str | Path] = None,
    progress_callback: Optional[callable] = None,
    use_reading_order: bool = True,
    primary_engine: Optional[str] = None,
    align_pages: bool = True,
) -> list[HarmonizedWord]:
    """
    Extract words from a PDF using smart harmonization with full tracking.

    Uses a configurable primary engine for word boundaries (default: tesseract).
    Secondary engines vote on text only and can add high-confidence detections.
    All detections are tracked with status and Levenshtein distances.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering pages
        use_tesseract: Whether to use Tesseract OCR
        use_easyocr: Whether to use EasyOCR
        use_paddleocr: Whether to use PaddleOCR (default: False)
        use_doctr: Whether to use docTR OCR (default: False)
        use_surya: Whether to use Surya OCR (default: True)
        use_pixel_detection: Whether to use pixel detection fallback
        gpu: Whether to use GPU for EasyOCR/PaddleOCR (default: False)
        preprocess: Preprocessing level - "none", "light", "standard", "aggressive", or "auto"
        upscale: Super-resolution scale factor (2, 4, or None for no upscaling)
        upscale_method: Super-resolution method - "espcn", "fsrcnn", "bicubic", "lanczos"
        tesseract_psm: Tesseract page segmentation mode (0-13, default 3)
        tesseract_oem: Tesseract OCR engine mode (0-3, default 3)
        easyocr_decoder: EasyOCR decoder - "greedy" or "beamsearch" (default: greedy)
        easyocr_text_threshold: EasyOCR text detection threshold (default: 0.7)
        config_path: Path to config file (uses default if None)
        progress_callback: Optional callback(page_num, total_pages, stage) for progress reporting
        use_reading_order: Use geometric clustering for proper reading order (default: True).
            When True, words are ordered by column-aware reading sequence.
            When False, words are ordered by simple (page, y, x) coordinates.
        primary_engine: Primary OCR engine for bbox authority ("tesseract", "surya", "doctr").
            If None, uses config setting (default: tesseract).
        align_pages: Auto-detect and correct page orientation (default: True).
            Uses Tesseract OSD to detect 90/180/270 degree rotations.

    Returns:
        List of HarmonizedWord with full tracking
    """
    pdf_path = Path(pdf_path)

    # Load config
    config = load_config(config_path) if config_path else get_config()

    # Determine primary engine (CLI override > config > default)
    primary = primary_engine or config.harmonize.primary.engine or "tesseract"

    # Check OCR availability
    tesseract_ok = is_tesseract_available() if use_tesseract else False
    easyocr_ok = is_easyocr_available() if use_easyocr else False
    paddleocr_ok = is_paddleocr_available() if use_paddleocr else False
    doctr_ok = is_doctr_available() if use_doctr else False
    surya_ok = is_surya_available() if use_surya else False

    # Map engine names to availability
    engine_available = {
        "tesseract": tesseract_ok,
        "easyocr": easyocr_ok,
        "paddleocr": paddleocr_ok,
        "doctr": doctr_ok,
        "surya": surya_ok,
    }

    # Validate primary engine is available
    if primary not in engine_available:
        raise RuntimeError(f"Unknown primary engine: {primary}")
    if not engine_available[primary]:
        raise RuntimeError(
            f"Primary engine '{primary}' is not available. Install it or choose another."
        )

    all_harmonized: list[HarmonizedWord] = []
    word_id_counter = 0

    with load_pdf(pdf_path, dpi=dpi) as pdf:
        total_pages = len(pdf)

        for page_num, image, page_width, page_height in pdf.pages():
            # Report progress: starting page
            if progress_callback:
                progress_callback(page_num, total_pages, "processing")

            # Store original page dimensions for coordinate transformation
            original_page_width = page_width
            original_page_height = page_height
            rotation_angle = 0

            # Page alignment: detect and correct rotation
            if align_pages and config.page_alignment.enabled:
                image, orientation = align_page(
                    image,
                    method=config.page_alignment.method,
                    min_confidence=config.page_alignment.min_confidence,
                    allowed_angles=config.page_alignment.angles,
                    surya_fallback_threshold=config.page_alignment.surya_fallback_threshold,
                )
                rotation_angle = orientation.angle
                # If rotated 90 or 270 degrees, dimensions swap for OCR
                if orientation.angle in [90, 270]:
                    page_width, page_height = page_height, page_width

            # Apply super-resolution if requested (before preprocessing)
            ocr_image = image
            if upscale and upscale > 1:
                ocr_image = upscale_image(ocr_image, scale=upscale, method=upscale_method)

            # Apply preprocessing if requested
            if preprocess and preprocess != "none":
                if preprocess == "auto":
                    level = auto_detect_quality(ocr_image)
                else:
                    level = PreprocessLevel(preprocess)
                ocr_image = preprocess_for_ocr(ocr_image, level=level, return_rgb=True)

            # Extract words from all enabled engines
            all_engine_results: dict[str, list[Word]] = {}
            tess_config = f"--psm {tesseract_psm} --oem {tesseract_oem}"

            if tesseract_ok:
                all_engine_results["tesseract"] = extract_words_tesseract(
                    ocr_image, page_num, page_width, page_height, config=tess_config
                )

            if easyocr_ok:
                all_engine_results["easyocr"] = extract_words_easyocr(
                    ocr_image, page_num, page_width, page_height, gpu=gpu,
                    decoder=easyocr_decoder, text_threshold=easyocr_text_threshold
                )

            if paddleocr_ok:
                all_engine_results["paddleocr"] = extract_words_paddleocr(
                    ocr_image, page_num, page_width, page_height, use_gpu=gpu
                )

            if doctr_ok:
                all_engine_results["doctr"] = extract_words_doctr(
                    ocr_image, page_num, page_width, page_height
                )

            if surya_ok:
                all_engine_results["surya"] = extract_words_surya(
                    ocr_image, page_num, page_width, page_height
                )

            # Split into primary and secondary based on config
            primary_words = all_engine_results.get(primary, [])
            secondary_results: dict[str, list[Word]] = {
                eng: words for eng, words in all_engine_results.items()
                if eng != primary
            }

            # Harmonize results
            page_harmonized = harmonize(primary_words, secondary_results, config, primary_engine=primary)

            # Pixel detection fallback for missed content
            if use_pixel_detection:
                existing_bboxes = [hw.bbox for hw in page_harmonized]
                pixel_words = detect_missed_content(
                    image, page_num, page_width, page_height,
                    existing_bboxes=existing_bboxes
                )

                # Convert pixel detections to HarmonizedWord
                for pw in pixel_words:
                    hw = HarmonizedWord(
                        word_id=-1,
                        page=pw.page,
                        bbox=pw.bbox,
                        text=pw.text,
                        status="pixel",
                        source="PX",  # PX for pixel detection
                        confidence=pw.confidence or 0,
                    )
                    page_harmonized.append(hw)

            # Store rotation angle on each word so the UI can rotate PDF display
            # Bboxes stay in rotated coordinate space (no transform back to original)
            for hw in page_harmonized:
                hw.rotation = rotation_angle

            # Assign word IDs
            for hw in page_harmonized:
                hw.word_id = word_id_counter
                word_id_counter += 1

            all_harmonized.extend(page_harmonized)

    # Order words for reading
    if use_reading_order:
        # Use geometric clustering for proper column-aware reading order
        all_harmonized = order_words_by_reading(all_harmonized)
    else:
        # Simple coordinate-based sorting (legacy behavior)
        all_harmonized.sort(key=lambda w: (w.page, w.bbox.y0, w.bbox.x0))
        # Reassign word IDs after sorting
        for i, hw in enumerate(all_harmonized):
            hw.word_id = i

    return all_harmonized


def extract_document(
    pdf_path: str | Path,
    dpi: int = 300,
    use_tesseract: bool = True,
    use_easyocr: bool = True,
    use_paddleocr: bool = False,
    use_doctr: bool = False,
    use_surya: bool = True,
    use_pixel_detection: bool = True,
    gpu: bool = False,
    preprocess: Optional[str] = "auto",
    upscale: Optional[int] = None,
    upscale_method: str = "espcn",
    triage: Optional[str] = None,
    tesseract_psm: int = 3,
    tesseract_oem: int = 3,
    easyocr_decoder: str = "greedy",
    easyocr_text_threshold: float = 0.7,
    config_path: Optional[str | Path] = None,
    progress_callback: Optional[callable] = None,
    use_reading_order: bool = True,
    primary_engine: Optional[str] = None,
    align_pages: bool = True,
) -> Document:
    """
    Extract words from a PDF and return as Document object.

    This is a convenience wrapper around extract_words() that returns
    a Document object for backward compatibility with APIs expecting
    the Document format.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for rendering pages
        use_tesseract: Whether to use Tesseract OCR (primary engine)
        use_easyocr: Whether to use EasyOCR
        use_paddleocr: Whether to use PaddleOCR (default: False)
        use_doctr: Whether to use docTR OCR (default: False)
        use_surya: Whether to use Surya OCR (default: False)
        use_pixel_detection: Whether to use pixel detection fallback
        gpu: Whether to use GPU for EasyOCR/PaddleOCR (default: False)
        preprocess: Preprocessing level - "none", "light", "standard", "aggressive", or "auto"
        upscale: Super-resolution scale factor (2, 4, or None for no upscaling)
        upscale_method: Super-resolution method - "espcn", "fsrcnn", "bicubic", "lanczos"
        triage: Triage level - "strict", "normal", "permissive", or None (no triage)
        tesseract_psm: Tesseract page segmentation mode (0-13, default 3)
        tesseract_oem: Tesseract OCR engine mode (0-3, default 3)
        easyocr_decoder: EasyOCR decoder - "greedy" or "beamsearch" (default: greedy)
        easyocr_text_threshold: EasyOCR text detection threshold (default: 0.7)
        config_path: Path to config file (uses default if None)
        progress_callback: Optional callback(page_num, total_pages, stage) for progress reporting
        use_reading_order: Use geometric clustering for proper reading order (default: True)

    Returns:
        Document with extracted words
    """
    pdf_path = Path(pdf_path)

    # Extract harmonized words
    harmonized_words = extract_words(
        pdf_path=pdf_path,
        dpi=dpi,
        use_tesseract=use_tesseract,
        use_easyocr=use_easyocr,
        use_paddleocr=use_paddleocr,
        use_doctr=use_doctr,
        use_surya=use_surya,
        use_pixel_detection=use_pixel_detection,
        gpu=gpu,
        preprocess=preprocess,
        upscale=upscale,
        upscale_method=upscale_method,
        tesseract_psm=tesseract_psm,
        tesseract_oem=tesseract_oem,
        easyocr_decoder=easyocr_decoder,
        easyocr_text_threshold=easyocr_text_threshold,
        config_path=config_path,
        progress_callback=progress_callback,
        use_reading_order=use_reading_order,
        primary_engine=primary_engine,
        align_pages=align_pages,
    )

    # Convert to Document
    doc = Document(filename=pdf_path.name)

    # Group words by page
    pages_dict: dict[int, list[Word]] = {}
    for hw in harmonized_words:
        if hw.page not in pages_dict:
            pages_dict[hw.page] = []

        # Convert HarmonizedWord to Word
        word = Word(
            word_id=hw.word_id,
            text=hw.text,
            bbox=hw.bbox,
            page=hw.page,
            engine=hw.source,
            confidence=hw.confidence,
        )
        pages_dict[hw.page].append(word)

    # Create Page objects (we need page dimensions from PDF)
    with load_pdf(pdf_path, dpi=dpi) as pdf:
        for page_num, _, page_width, page_height in pdf.pages():
            page = Page(
                page_number=page_num,
                width=page_width,
                height=page_height,
            )
            page.words = pages_dict.get(page_num, [])
            doc.pages.append(page)

    # Apply triage filtering if requested
    if triage:
        triage_level = TriageLevel(triage)
        for page in doc.pages:
            page.words = triage_words(page.words, level=triage_level)

        # Reassign word IDs after triage
        word_id_counter = 0
        for page in doc.pages:
            for word in page.words:
                word.word_id = word_id_counter
                word_id_counter += 1

    return doc


def extract_to_csv(
    pdf_path: str | Path,
    output_path: str | Path,
    dpi: int = 300,
    use_easyocr: bool = True,
    use_paddleocr: bool = False,
    use_doctr: bool = False,
    preprocess: Optional[str] = "none",
    tesseract_psm: int = 6,
    tesseract_oem: int = 3,
) -> list[HarmonizedWord]:
    """
    Extract words from PDF and write to CSV.

    Args:
        pdf_path: Path to input PDF
        output_path: Path to output CSV
        dpi: Resolution for rendering
        use_easyocr: Whether to use EasyOCR
        use_paddleocr: Whether to use PaddleOCR
        use_doctr: Whether to use docTR
        preprocess: Preprocessing level
        tesseract_psm: Tesseract page segmentation mode
        tesseract_oem: Tesseract OCR engine mode

    Returns:
        List of HarmonizedWord
    """
    from .output import write_harmonized_csv

    words = extract_words(
        pdf_path,
        dpi=dpi,
        use_easyocr=use_easyocr,
        use_paddleocr=use_paddleocr,
        use_doctr=use_doctr,
        preprocess=preprocess,
        tesseract_psm=tesseract_psm,
        tesseract_oem=tesseract_oem,
    )
    write_harmonized_csv(words, output_path)
    return words


# Backward compatibility alias
extract_words_smart = extract_words
