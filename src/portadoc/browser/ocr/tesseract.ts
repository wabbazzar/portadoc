/**
 * Tesseract.js wrapper for browser-based OCR.
 */

import Tesseract from 'tesseract.js';
import { Word, BBox } from '../models';
import { adjustConfidence } from '../config';

let worker: Tesseract.Worker | null = null;

/**
 * Initialize or get the Tesseract worker.
 */
async function getWorker(onProgress?: (progress: number) => void): Promise<Tesseract.Worker> {
  if (worker) {
    console.log('[Tesseract] Reusing existing worker');
    return worker;
  }

  console.log('[Tesseract] Creating new worker...');
  try {
    worker = await Tesseract.createWorker('eng', 1, {
      logger: (m) => {
        console.log('[Tesseract Worker]', m.status, m.progress ? `${Math.round(m.progress * 100)}%` : '');
        if (m.status === 'recognizing text' && onProgress) {
          onProgress(m.progress);
        }
      },
    });
    console.log('[Tesseract] Worker created successfully');
  } catch (error) {
    console.error('[Tesseract] FAILED to create worker:', error);
    throw error;
  }

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
  console.log('[Tesseract] Getting worker...');
  const w = await getWorker(onProgress);

  console.log('[Tesseract] Running recognize...');
  const result = await w.recognize(imageData);

  console.log('[Tesseract] Raw result:', {
    text: result.data.text?.substring(0, 200),
    wordCount: result.data.words?.length,
    confidence: result.data.confidence,
  });

  const words: Word[] = [];

  if (result.data.words) {
    console.log('[Tesseract] Processing', result.data.words.length, 'raw words');
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
        confidence: adjustConfidence('tesseract', word.confidence),
      });
    }
    console.log('[Tesseract] After filtering:', words.length, 'words');
  } else {
    console.log('[Tesseract] WARNING: No words in result.data.words!');
    console.log('[Tesseract] Full result.data:', JSON.stringify(result.data).substring(0, 500));
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
