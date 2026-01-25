/**
 * Tesseract.js wrapper for browser-based OCR.
 */

import Tesseract from 'tesseract.js';
import { Word, BBox } from '../models';

let worker: Tesseract.Worker | null = null;

/**
 * Initialize or get the Tesseract worker.
 */
async function getWorker(onProgress?: (progress: number) => void): Promise<Tesseract.Worker> {
  if (worker) {
    return worker;
  }

  worker = await Tesseract.createWorker('eng', 1, {
    logger: (m) => {
      if (m.status === 'recognizing text' && onProgress) {
        onProgress(m.progress);
      }
    },
  });

  return worker;
}

/**
 * Extract words from an image using Tesseract.js.
 * @param imageData Image as data URL or ImageData
 * @param pageIndex Page number for result
 * @param onProgress Progress callback (0-1)
 * @returns Array of extracted words
 */
export async function extractWithTesseract(
  imageData: string | ImageData,
  pageIndex: number,
  onProgress?: (progress: number) => void
): Promise<Word[]> {
  const w = await getWorker(onProgress);

  const result = await w.recognize(imageData);

  const words: Word[] = [];

  if (result.data.words) {
    for (const word of result.data.words) {
      // Skip empty words
      if (!word.text.trim()) continue;

      const bbox: BBox = {
        x0: word.bbox.x0,
        y0: word.bbox.y0,
        x1: word.bbox.x1,
        y1: word.bbox.y1,
      };

      words.push({
        page: pageIndex,
        wordId: words.length,
        text: word.text,
        bbox,
        engine: 'tesseract',
        confidence: word.confidence,
      });
    }
  }

  return words;
}

/**
 * Terminate the worker to free resources.
 */
export async function terminateTesseract(): Promise<void> {
  if (worker) {
    await worker.terminate();
    worker = null;
  }
}
