/**
 * PDF.js wrapper for loading and rendering PDF files.
 */

import * as pdfjsLib from 'pdfjs-dist';

// Set worker source - use CDN for simplicity
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

export type PdfDocument = pdfjsLib.PDFDocumentProxy;

/**
 * Load a PDF file and return the document proxy.
 */
export async function loadPdf(file: File): Promise<PdfDocument> {
  const arrayBuffer = await file.arrayBuffer();
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
  return loadingTask.promise;
}

/**
 * Get the number of pages in a PDF document.
 */
export function getPdfPageCount(doc: PdfDocument): number {
  return doc.numPages;
}

/**
 * Render a specific page to a canvas.
 * @param doc PDF document
 * @param pageIndex 0-based page index
 * @param canvas Target canvas element
 * @param scale Render scale (default 1.5 for good quality)
 */
export async function renderPage(
  doc: PdfDocument,
  pageIndex: number,
  canvas: HTMLCanvasElement,
  scale: number = 1.5
): Promise<void> {
  // PDF.js uses 1-based page numbers
  const page = await doc.getPage(pageIndex + 1);
  const viewport = page.getViewport({ scale });

  canvas.width = viewport.width;
  canvas.height = viewport.height;

  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const renderContext = {
    canvasContext: ctx,
    viewport: viewport,
  };

  await page.render(renderContext).promise;
}

/**
 * Get page dimensions without rendering.
 */
export async function getPageDimensions(
  doc: PdfDocument,
  pageIndex: number,
  scale: number = 1.5
): Promise<{ width: number; height: number }> {
  const page = await doc.getPage(pageIndex + 1);
  const viewport = page.getViewport({ scale });
  return { width: viewport.width, height: viewport.height };
}

/**
 * Get the raw bytes of a PDF file.
 * Used for exporting modified PDFs.
 */
export async function getPdfBytes(file: File): Promise<ArrayBuffer> {
  return file.arrayBuffer();
}
