"""Configuration loader for Portadoc harmonization and OCR settings."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class StatusConfig:
    """Status threshold configuration."""
    word_min_conf: float = 60.0
    low_conf_min_conf: float = 20.0


@dataclass
class EngineConfig:
    """Per-engine configuration."""
    enabled: bool = True
    weight: float = 1.0
    garbage_penalty: float = 0.1
    bbox_type: str = "word"  # "word" or "line"
    # Per-engine matching overrides (for engines with offset bboxes like Surya)
    iou_threshold_override: Optional[float] = None
    center_distance_max_override: Optional[float] = None


@dataclass
class SecondaryConfig:
    """Secondary engine configuration."""
    vote_min_conf: float = 40.0
    solo_min_conf: float = 85.0
    solo_high_conf: float = 95.0
    engines: dict[str, EngineConfig] = field(default_factory=dict)


@dataclass
class PrimaryConfig:
    """Primary engine configuration."""
    engine: str = "tesseract"
    weight: float = 1.0


@dataclass
class GarbageDetectionConfig:
    """Garbage text detection configuration."""
    enabled: bool = True
    min_alnum_ratio: float = 0.6
    max_consonant_run: int = 4
    mixed_case_penalty: bool = True


@dataclass
class HarmonizeConfig:
    """Harmonization configuration."""
    iou_threshold: float = 0.3
    text_match_bonus: float = 0.15  # Lower IoU threshold when text matches
    center_distance_max: float = 12.0  # Max center distance for fallback matching
    status: StatusConfig = field(default_factory=StatusConfig)
    primary: PrimaryConfig = field(default_factory=PrimaryConfig)
    secondary: SecondaryConfig = field(default_factory=SecondaryConfig)
    garbage_detection: GarbageDetectionConfig = field(default_factory=GarbageDetectionConfig)


@dataclass
class TesseractConfig:
    """Tesseract OCR configuration."""
    psm: int = 6
    oem: int = 3


@dataclass
class EasyOCRConfig:
    """EasyOCR configuration."""
    decoder: str = "greedy"
    beamWidth: int = 5
    text_threshold: float = 0.7
    low_text: float = 0.4
    contrast_ths: float = 0.1
    adjust_contrast: float = 0.5
    width_ths: float = 0.5
    mag_ratio: float = 1.0


@dataclass
class PaddleOCRConfig:
    """PaddleOCR configuration."""
    use_angle_cls: bool = True
    lang: str = "en"
    use_gpu: bool = False


@dataclass
class DocTRConfig:
    """docTR configuration."""
    pretrained: bool = True
    assume_straight_pages: bool = True


@dataclass
class OCRConfig:
    """OCR engine configurations."""
    tesseract: TesseractConfig = field(default_factory=TesseractConfig)
    easyocr: EasyOCRConfig = field(default_factory=EasyOCRConfig)
    paddleocr: PaddleOCRConfig = field(default_factory=PaddleOCRConfig)
    doctr: DocTRConfig = field(default_factory=DocTRConfig)


@dataclass
class CSVOutputConfig:
    """CSV output configuration."""
    include_all_engines: bool = True
    include_distances: bool = True
    include_pixel_detections: bool = True


@dataclass
class OutputConfig:
    """Output configuration."""
    csv: CSVOutputConfig = field(default_factory=CSVOutputConfig)


@dataclass
class PageAlignmentConfig:
    """Page alignment (auto-rotation) configuration."""
    enabled: bool = True
    method: str = "auto"  # tesseract_osd, surya, or auto (OSD with Surya fallback)
    min_confidence: float = 0.1  # OSD confidence is typically low (0.1-0.3) even for correct detections
    angles: list[int] = field(default_factory=lambda: [90, 180, 270])
    surya_fallback_threshold: float = 0.05  # Use Surya when OSD confidence below this


@dataclass
class PortadocConfig:
    """Root configuration object."""
    harmonize: HarmonizeConfig = field(default_factory=HarmonizeConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    page_alignment: PageAlignmentConfig = field(default_factory=PageAlignmentConfig)


def _parse_engine_config(data: dict) -> EngineConfig:
    """Parse engine configuration from dict."""
    return EngineConfig(
        enabled=data.get("enabled", True),
        weight=data.get("weight", 1.0),
        garbage_penalty=data.get("garbage_penalty", 0.1),
        bbox_type=data.get("bbox_type", "word"),
        iou_threshold_override=data.get("iou_threshold_override"),
        center_distance_max_override=data.get("center_distance_max_override"),
    )


def load_config(config_path: Optional[Path | str] = None) -> PortadocConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default config/harmonize.yaml

    Returns:
        PortadocConfig object with all settings
    """
    if config_path is None:
        # Try default location
        default_path = Path(__file__).parent.parent.parent / "config" / "harmonize.yaml"
        if default_path.exists():
            config_path = default_path
        else:
            # Return defaults
            return PortadocConfig()

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    return _parse_config(data)


def _parse_config(data: dict) -> PortadocConfig:
    """Parse configuration from dict."""
    config = PortadocConfig()

    # Parse harmonize section
    if "harmonize" in data:
        h = data["harmonize"]
        config.harmonize.iou_threshold = h.get("iou_threshold", 0.3)
        config.harmonize.text_match_bonus = h.get("text_match_bonus", 0.15)
        config.harmonize.center_distance_max = h.get("center_distance_max", 12.0)

        # Status
        if "status" in h:
            s = h["status"]
            config.harmonize.status.word_min_conf = s.get("word_min_conf", 60.0)
            config.harmonize.status.low_conf_min_conf = s.get("low_conf_min_conf", 20.0)

        # Primary
        if "primary" in h:
            p = h["primary"]
            config.harmonize.primary.engine = p.get("engine", "tesseract")
            config.harmonize.primary.weight = p.get("weight", 1.0)

        # Secondary
        if "secondary" in h:
            s = h["secondary"]
            config.harmonize.secondary.vote_min_conf = s.get("vote_min_conf", 40.0)
            config.harmonize.secondary.solo_min_conf = s.get("solo_min_conf", 85.0)
            config.harmonize.secondary.solo_high_conf = s.get("solo_high_conf", 95.0)

            if "engines" in s:
                for name, eng_data in s["engines"].items():
                    config.harmonize.secondary.engines[name] = _parse_engine_config(eng_data)

        # Garbage detection
        if "garbage_detection" in h:
            g = h["garbage_detection"]
            config.harmonize.garbage_detection.enabled = g.get("enabled", True)
            config.harmonize.garbage_detection.min_alnum_ratio = g.get("min_alnum_ratio", 0.6)
            config.harmonize.garbage_detection.max_consonant_run = g.get("max_consonant_run", 4)
            config.harmonize.garbage_detection.mixed_case_penalty = g.get("mixed_case_penalty", True)

    # Parse OCR section
    if "ocr" in data:
        o = data["ocr"]

        if "tesseract" in o:
            t = o["tesseract"]
            config.ocr.tesseract.psm = t.get("psm", 6)
            config.ocr.tesseract.oem = t.get("oem", 3)

        if "easyocr" in o:
            e = o["easyocr"]
            config.ocr.easyocr.decoder = e.get("decoder", "greedy")
            config.ocr.easyocr.beamWidth = e.get("beamWidth", 5)
            config.ocr.easyocr.text_threshold = e.get("text_threshold", 0.7)
            config.ocr.easyocr.low_text = e.get("low_text", 0.4)
            config.ocr.easyocr.contrast_ths = e.get("contrast_ths", 0.1)
            config.ocr.easyocr.adjust_contrast = e.get("adjust_contrast", 0.5)
            config.ocr.easyocr.width_ths = e.get("width_ths", 0.5)
            config.ocr.easyocr.mag_ratio = e.get("mag_ratio", 1.0)

        if "paddleocr" in o:
            p = o["paddleocr"]
            config.ocr.paddleocr.use_angle_cls = p.get("use_angle_cls", True)
            config.ocr.paddleocr.lang = p.get("lang", "en")
            config.ocr.paddleocr.use_gpu = p.get("use_gpu", False)

        if "doctr" in o:
            d = o["doctr"]
            config.ocr.doctr.pretrained = d.get("pretrained", True)
            config.ocr.doctr.assume_straight_pages = d.get("assume_straight_pages", True)

    # Parse output section
    if "output" in data:
        o = data["output"]
        if "csv" in o:
            c = o["csv"]
            config.output.csv.include_all_engines = c.get("include_all_engines", True)
            config.output.csv.include_distances = c.get("include_distances", True)
            config.output.csv.include_pixel_detections = c.get("include_pixel_detections", True)

    # Parse page_alignment section
    if "page_alignment" in data:
        pa = data["page_alignment"]
        config.page_alignment.enabled = pa.get("enabled", True)
        config.page_alignment.method = pa.get("method", "auto")
        config.page_alignment.min_confidence = pa.get("min_confidence", 0.1)
        config.page_alignment.angles = pa.get("angles", [90, 180, 270])
        config.page_alignment.surya_fallback_threshold = pa.get("surya_fallback_threshold", 0.05)

    return config


# Global config instance (lazy loaded)
_config: Optional[PortadocConfig] = None


def get_config(config_path: Optional[Path | str] = None) -> PortadocConfig:
    """
    Get the global configuration instance.

    Args:
        config_path: If provided, reload config from this path

    Returns:
        PortadocConfig instance
    """
    global _config
    if _config is None or config_path is not None:
        _config = load_config(config_path)
    return _config
