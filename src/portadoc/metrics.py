"""Evaluation metrics for OCR output vs ground truth."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .models import BBox, Document, Word, HarmonizedWord


@dataclass
class MatchResult:
    """Result of matching a predicted word to ground truth."""

    pred_word: Word
    gt_word: Optional[Word]
    iou: float
    text_match: bool


@dataclass
class EvaluationResult:
    """Complete evaluation metrics."""

    # Counts
    total_predicted: int
    total_ground_truth: int
    true_positives: int  # Matched by bbox IoU
    false_positives: int  # Predicted but no GT match
    false_negatives: int  # GT but no prediction match

    # Core metrics
    precision: float  # TP / (TP + FP)
    recall: float  # TP / (TP + FN)
    f1_score: float  # Harmonic mean of precision and recall

    # IoU metrics
    mean_iou: float  # Average IoU of matched pairs
    matched_text_ratio: float  # % of matches with exact text

    # Details
    matches: list[MatchResult]
    unmatched_gt: list[Word]
    false_positive_words: list[Word]

    def summary(self) -> str:
        """Return human-readable summary."""
        return f"""Evaluation Results:
  Ground Truth: {self.total_ground_truth} words
  Predicted:    {self.total_predicted} words

  True Positives:  {self.true_positives}
  False Positives: {self.false_positives}
  False Negatives: {self.false_negatives}

  Precision: {self.precision:.2%}
  Recall:    {self.recall:.2%}
  F1 Score:  {self.f1_score:.2%}

  Mean IoU (matched): {self.mean_iou:.4f}
  Text Match Rate:    {self.matched_text_ratio:.2%}"""


def load_ground_truth_csv(csv_path: Path | str) -> list[Word]:
    """
    Load ground truth words from CSV.

    Expected format: page,word_id,text,x0,y0,x1,y1,engine,ocr_confidence
    """
    csv_path = Path(csv_path)
    words = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = Word(
                word_id=int(row["word_id"]),
                text=row["text"],
                bbox=BBox(
                    x0=float(row["x0"]),
                    y0=float(row["y0"]),
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                ),
                page=int(row["page"]),
                engine=row.get("engine", ""),
                confidence=float(row.get("ocr_confidence", 0.0) or 0.0),
            )
            words.append(word)

    return words


def match_words(
    predicted: list[Word],
    ground_truth: list[Word],
    iou_threshold: float = 0.5,
) -> tuple[list[MatchResult], list[Word], list[Word]]:
    """
    Match predicted words to ground truth by bounding box IoU.

    Args:
        predicted: Predicted words from extraction
        ground_truth: Ground truth words
        iou_threshold: Minimum IoU to consider a match

    Returns:
        Tuple of (matches, unmatched_gt, false_positives)
    """
    matches = []
    used_gt = set()
    false_positives = []

    # Group by page for efficiency
    gt_by_page: dict[int, list[Word]] = {}
    for w in ground_truth:
        gt_by_page.setdefault(w.page, []).append(w)

    for pred in predicted:
        page_gt = gt_by_page.get(pred.page, [])

        best_match = None
        best_iou = 0.0

        for gt in page_gt:
            if id(gt) in used_gt:
                continue

            iou = pred.bbox.iou(gt.bbox)
            if iou > best_iou and iou >= iou_threshold:
                best_iou = iou
                best_match = gt

        if best_match:
            used_gt.add(id(best_match))
            text_match = normalize_text(pred.text) == normalize_text(best_match.text)
            matches.append(MatchResult(
                pred_word=pred,
                gt_word=best_match,
                iou=best_iou,
                text_match=text_match,
            ))
        else:
            false_positives.append(pred)

    # Find unmatched ground truth
    unmatched_gt = [gt for gt in ground_truth if id(gt) not in used_gt]

    return matches, unmatched_gt, false_positives


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.lower().strip()


def harmonized_to_words(harmonized: list[HarmonizedWord]) -> list[Word]:
    """Convert HarmonizedWord list to Word list for evaluation."""
    return [
        Word(
            word_id=hw.word_id,
            text=hw.text,
            bbox=hw.bbox,
            page=hw.page,
            engine=hw.source,
            confidence=hw.confidence,
        )
        for hw in harmonized
    ]


def evaluate(
    predicted: list[Word] | list[HarmonizedWord] | Document,
    ground_truth: list[Word] | Path | str,
    iou_threshold: float = 0.5,
) -> EvaluationResult:
    """
    Evaluate predicted words against ground truth.

    Args:
        predicted: Predicted words, HarmonizedWords, or Document
        ground_truth: Ground truth words or path to CSV
        iou_threshold: Minimum IoU for match

    Returns:
        EvaluationResult with all metrics
    """
    # Handle Document input
    if isinstance(predicted, Document):
        predicted = predicted.all_words()

    # Handle HarmonizedWord list
    if predicted and isinstance(predicted[0], HarmonizedWord):
        predicted = harmonized_to_words(predicted)

    # Handle CSV path input
    if isinstance(ground_truth, (str, Path)):
        ground_truth = load_ground_truth_csv(ground_truth)

    # Match predictions to ground truth
    matches, unmatched_gt, false_positives = match_words(
        predicted, ground_truth, iou_threshold
    )

    # Calculate metrics
    tp = len(matches)
    fp = len(false_positives)
    fn = len(unmatched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mean_iou = sum(m.iou for m in matches) / len(matches) if matches else 0.0
    text_matches = sum(1 for m in matches if m.text_match)
    text_ratio = text_matches / len(matches) if matches else 0.0

    return EvaluationResult(
        total_predicted=len(predicted),
        total_ground_truth=len(ground_truth),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=precision,
        recall=recall,
        f1_score=f1,
        mean_iou=mean_iou,
        matched_text_ratio=text_ratio,
        matches=matches,
        unmatched_gt=unmatched_gt,
        false_positive_words=false_positives,
    )
