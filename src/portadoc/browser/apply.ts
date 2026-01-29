/**
 * Apply redactions to PDFs in the browser.
 *
 * Provides:
 * - Visual preview overlay on canvas
 * - Full PDF export with redactions (using pdf-lib)
 */

import { PDFDocument, rgb } from 'pdf-lib';
import { ExtractedWord } from './models';
import { EntityType } from './redact';

export interface RedactionBox {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface RenderOptions {
  /** Color for redaction boxes (CSS color string) */
  color?: string;
  /** Opacity of the fill (0-1) */
  fillOpacity?: number;
  /** Whether to show border */
  showBorder?: boolean;
  /** Border color (CSS color string) */
  borderColor?: string;
  /** Border width in pixels */
  borderWidth?: number;
}

const DEFAULT_RENDER_OPTIONS: RenderOptions = {
  color: '#000000',
  fillOpacity: 0.8,
  showBorder: true,
  borderColor: '#ff0000',
  borderWidth: 2,
};

/**
 * Render redaction overlay boxes on top of the PDF canvas.
 *
 * Creates semi-transparent boxes over words marked for redaction.
 * Used for preview before final PDF export.
 *
 * @param container - The DOM element to render boxes into (should overlay the canvas)
 * @param words - Array of ExtractedWord with entity and redact fields
 * @param pageIndex - Current page index (0-based)
 * @param options - Rendering options
 * @returns Array of created overlay elements (for cleanup)
 */
export function renderRedactionOverlay(
  container: HTMLElement,
  words: ExtractedWord[],
  pageIndex: number,
  options: RenderOptions = {}
): HTMLDivElement[] {
  const opts = { ...DEFAULT_RENDER_OPTIONS, ...options };
  const elements: HTMLDivElement[] = [];

  // Filter words for current page that are marked for redaction
  const redactWords = words.filter(
    w => w.page === pageIndex && w.redact === true
  );

  for (const word of redactWords) {
    const div = document.createElement('div');
    div.className = 'redaction-overlay-box';

    // Position and size
    div.style.position = 'absolute';
    div.style.left = `${word.bbox.x0}px`;
    div.style.top = `${word.bbox.y0}px`;
    div.style.width = `${word.bbox.x1 - word.bbox.x0}px`;
    div.style.height = `${word.bbox.y1 - word.bbox.y0}px`;

    // Styling based on entity type
    const entityColor = getEntityColor(word.entity);
    div.style.backgroundColor = entityColor;
    div.style.opacity = String(opts.fillOpacity);

    if (opts.showBorder) {
      div.style.border = `${opts.borderWidth}px solid ${opts.borderColor}`;
      div.style.boxSizing = 'border-box';
    }

    // Pointer events pass through
    div.style.pointerEvents = 'none';

    // Store word ID for reference
    div.dataset.wordId = String(word.wordId);
    div.dataset.entity = word.entity || '';

    container.appendChild(div);
    elements.push(div);
  }

  return elements;
}

/**
 * Clear all redaction overlay boxes from a container.
 */
export function clearRedactionOverlay(container: HTMLElement): void {
  const boxes = container.querySelectorAll('.redaction-overlay-box');
  boxes.forEach(box => box.remove());
}

/**
 * Get color for an entity type.
 */
function getEntityColor(entity?: EntityType | string): string {
  switch (entity) {
    case EntityType.NAME:
    case 'NAME':
      return '#ff6b6b'; // Red for names
    case EntityType.DATE:
    case 'DATE':
      return '#4dabf7'; // Blue for dates
    case EntityType.CODE:
    case 'CODE':
      return '#51cf66'; // Green for codes
    default:
      return '#868e96'; // Gray for unknown
  }
}

/**
 * Export a redacted PDF with black boxes over marked words.
 *
 * Uses pdf-lib to modify the PDF document directly.
 *
 * @param pdfBytes - Original PDF as ArrayBuffer
 * @param redactions - Array of redaction boxes
 * @param color - RGB color tuple (0-1 range) for redaction fill
 * @returns Modified PDF as Uint8Array
 */
export async function exportRedactedPdf(
  pdfBytes: ArrayBuffer,
  redactions: RedactionBox[],
  color: [number, number, number] = [0, 0, 0]
): Promise<Uint8Array> {
  // Load the PDF
  const pdfDoc = await PDFDocument.load(pdfBytes);
  const pages = pdfDoc.getPages();

  // Group redactions by page
  const redactionsByPage = new Map<number, RedactionBox[]>();
  for (const redaction of redactions) {
    const list = redactionsByPage.get(redaction.page) || [];
    list.push(redaction);
    redactionsByPage.set(redaction.page, list);
  }

  // Apply redactions to each page
  for (const [pageIndex, pageRedactions] of redactionsByPage) {
    if (pageIndex >= pages.length) continue;

    const page = pages[pageIndex];
    const { height } = page.getSize();

    for (const box of pageRedactions) {
      // pdf-lib uses bottom-left origin, so flip y coordinates
      // The bounding boxes from OCR use top-left origin
      const y = height - box.y1; // Convert to bottom-left origin

      page.drawRectangle({
        x: box.x0,
        y: y,
        width: box.x1 - box.x0,
        height: box.y1 - box.y0,
        color: rgb(color[0], color[1], color[2]),
      });
    }
  }

  // Save and return the modified PDF
  return pdfDoc.save();
}

/**
 * Export redacted PDF from ExtractedWord array.
 *
 * Convenience function that extracts redaction boxes from words.
 *
 * @param pdfBytes - Original PDF as ArrayBuffer
 * @param words - Array of ExtractedWord with redact field
 * @param color - RGB color tuple (0-1 range)
 * @returns Modified PDF as Uint8Array
 */
export async function exportRedactedPdfFromWords(
  pdfBytes: ArrayBuffer,
  words: ExtractedWord[],
  color: [number, number, number] = [0, 0, 0]
): Promise<Uint8Array> {
  // Extract redaction boxes from words marked for redaction
  const redactions: RedactionBox[] = words
    .filter(w => w.redact === true)
    .map(w => ({
      page: w.page,
      x0: w.bbox.x0,
      y0: w.bbox.y0,
      x1: w.bbox.x1,
      y1: w.bbox.y1,
    }));

  return exportRedactedPdf(pdfBytes, redactions, color);
}

/**
 * Download a Uint8Array as a file.
 */
export function downloadPdf(data: Uint8Array, filename: string): void {
  // Create a new ArrayBuffer copy to avoid SharedArrayBuffer compatibility issues
  const blob = new Blob([new Uint8Array(data)], { type: 'application/pdf' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url);
}

/**
 * Get redaction statistics from words array.
 */
export function getRedactionStats(words: ExtractedWord[]): {
  total: number;
  redacted: number;
  byType: Record<string, number>;
} {
  const stats = {
    total: words.length,
    redacted: 0,
    byType: {
      NAME: 0,
      DATE: 0,
      CODE: 0,
    } as Record<string, number>,
  };

  for (const word of words) {
    if (word.redact) {
      stats.redacted++;
      const type = word.entity || '';
      if (type in stats.byType) {
        stats.byType[type]++;
      }
    }
  }

  return stats;
}
