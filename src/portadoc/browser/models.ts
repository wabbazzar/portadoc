/**
 * Data models for Portadoc Browser client.
 * Port of Python models from src/portadoc/models.py
 */

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Word {
  page: number;
  wordId: number;
  text: string;
  bbox: BBox;
  engine: string;
  confidence: number;
}

export interface ExtractedWord extends Word {
  // Additional fields for harmonization
  matched?: boolean;
  sources?: string[];
}

export interface OcrResult {
  words: Word[];
  engine: string;
  pageIndex: number;
}

export interface HarmonizedResult {
  words: ExtractedWord[];
  pageIndex: number;
  tesseractCount: number;
  doctrCount: number;
  harmonizedCount: number;
}

// Helper functions for BBox operations
export function bboxWidth(bbox: BBox): number {
  return bbox.x1 - bbox.x0;
}

export function bboxHeight(bbox: BBox): number {
  return bbox.y1 - bbox.y0;
}

export function bboxCenterX(bbox: BBox): number {
  return (bbox.x0 + bbox.x1) / 2;
}

export function bboxCenterY(bbox: BBox): number {
  return (bbox.y0 + bbox.y1) / 2;
}

export function bboxArea(bbox: BBox): number {
  return bboxWidth(bbox) * bboxHeight(bbox);
}

export function bboxIoU(a: BBox, b: BBox): number {
  const x0 = Math.max(a.x0, b.x0);
  const y0 = Math.max(a.y0, b.y0);
  const x1 = Math.min(a.x1, b.x1);
  const y1 = Math.min(a.y1, b.y1);

  if (x1 <= x0 || y1 <= y0) {
    return 0;
  }

  const intersection = (x1 - x0) * (y1 - y0);
  const union = bboxArea(a) + bboxArea(b) - intersection;

  return union > 0 ? intersection / union : 0;
}

export function horizontalGap(a: BBox, b: BBox): number {
  if (a.x1 <= b.x0) {
    return b.x0 - a.x1;
  } else if (b.x1 <= a.x0) {
    return a.x0 - b.x1;
  }
  return 0; // Overlapping
}

export function verticalGap(a: BBox, b: BBox): number {
  if (a.y1 <= b.y0) {
    return b.y0 - a.y1;
  } else if (b.y1 <= a.y0) {
    return a.y0 - b.y1;
  }
  return 0; // Overlapping
}

export function xOverlapRatio(a: BBox, b: BBox): number {
  const overlapX0 = Math.max(a.x0, b.x0);
  const overlapX1 = Math.min(a.x1, b.x1);
  const overlapWidth = Math.max(0, overlapX1 - overlapX0);
  const minWidth = Math.min(bboxWidth(a), bboxWidth(b));
  return minWidth > 0 ? overlapWidth / minWidth : 0;
}

export function yOverlapRatio(a: BBox, b: BBox): number {
  const overlapY0 = Math.max(a.y0, b.y0);
  const overlapY1 = Math.min(a.y1, b.y1);
  const overlapHeight = Math.max(0, overlapY1 - overlapY0);
  const minHeight = Math.min(bboxHeight(a), bboxHeight(b));
  return minHeight > 0 ? overlapHeight / minHeight : 0;
}
