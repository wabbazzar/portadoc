/**
 * Pixel-based detection for OCR-missed content.
 * Port of Python detection.py using Canvas API.
 *
 * Detects:
 * 1. Logo/image regions (large, boxy shapes)
 * 2. Horizontal lines (separators, underlines)
 * 3. Vertical lines (margins, borders)
 */

import { BBox, Word, bboxIoU, bboxCenterX, bboxCenterY } from './models';

// Detection thresholds (matching Python detection.py)
const THRESHOLDS = {
  // Overlap detection
  overlapIoU: 0.1,

  // Logo detection
  logoMinAreaPts: 1000, // Minimum area in PDF points²
  logoMinDimensionPts: 30, // Minimum width/height in pts
  logoMaxAspectRatio: 20, // Skip line-like shapes

  // Text region detection (for OCR-missed text)
  textMinAreaPts: 50, // Minimum area - small text regions
  textMinWidthPts: 10, // Minimum width
  textMinHeightPts: 5, // Minimum height
  textMaxHeightPts: 50, // Maximum height (text isn't huge)
  textMaxAspectRatio: 50, // Text can be quite wide relative to height

  // Horizontal lines
  hLineMinWidthPts: 100,
  hLineMaxHeightPts: 10,
  hLineMinAspectRatio: 10,

  // Vertical lines
  vLineMinHeightPts: 50,
  vLineMaxWidthPts: 10,
  vLineMinAspectRatio: 5,

  // Binary threshold (grayscale value)
  binaryThreshold: 200,

  // Morphology
  dilateKernelSize: 5,
  dilateIterations: 3,

  // Text region morphology (smaller kernel for fine text)
  textDilateKernelSize: 3,
  textDilateIterations: 2,
};

/**
 * Simple 2D grayscale image representation.
 */
interface GrayImage {
  data: Uint8Array;
  width: number;
  height: number;
}

/**
 * Detected contour bounding box in image coordinates.
 */
interface ContourBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

// ============================================================================
// Yielding / Throttling
// ============================================================================

/**
 * Yield to the main thread to prevent UI freezes.
 * Uses scheduler.yield() if available, otherwise setTimeout.
 */
function yieldToMain(): Promise<void> {
  // @ts-ignore - scheduler.yield is a new API
  if (typeof scheduler !== 'undefined' && scheduler.yield) {
    // @ts-ignore
    return scheduler.yield();
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * Yield every N iterations to prevent blocking.
 */
const YIELD_INTERVAL = 50000; // Yield every 50k pixels processed

// ============================================================================
// Image Processing Primitives
// ============================================================================

/**
 * Convert RGBA ImageData to grayscale (async with yielding).
 */
async function toGrayscale(imageData: ImageData): Promise<GrayImage> {
  const { width, height, data } = imageData;
  const gray = new Uint8Array(width * height);
  const total = width * height;

  for (let i = 0; i < total; i++) {
    const r = data[i * 4];
    const g = data[i * 4 + 1];
    const b = data[i * 4 + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);

    // Yield periodically to prevent UI freeze
    if (i > 0 && i % YIELD_INTERVAL === 0) {
      await yieldToMain();
    }
  }

  return { data: gray, width, height };
}

/**
 * Apply binary threshold (inverse): pixels darker than threshold become white (255).
 * This matches cv2.THRESH_BINARY_INV behavior.
 */
function binaryThresholdInv(gray: GrayImage, threshold: number): GrayImage {
  const result = new Uint8Array(gray.data.length);

  for (let i = 0; i < gray.data.length; i++) {
    // THRESH_BINARY_INV: if pixel < threshold, output 255, else 0
    // But OpenCV uses > for THRESH_BINARY, so for INV it's <=
    // Actually cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV) means:
    // if gray[i] > 200: 0 else 255
    result[i] = gray.data[i] > threshold ? 0 : 255;
  }

  return { data: result, width: gray.width, height: gray.height };
}

/**
 * Dilate binary image with rectangular kernel (async with yielding).
 * A pixel becomes white if ANY pixel in the kernel neighborhood is white.
 */
async function dilate(
  binary: GrayImage,
  kernelWidth: number,
  kernelHeight: number,
  iterations: number = 1
): Promise<GrayImage> {
  let current = binary;

  for (let iter = 0; iter < iterations; iter++) {
    const result = new Uint8Array(current.data.length);
    const halfW = Math.floor(kernelWidth / 2);
    const halfH = Math.floor(kernelHeight / 2);
    let pixelCount = 0;

    for (let y = 0; y < current.height; y++) {
      for (let x = 0; x < current.width; x++) {
        let maxVal = 0;

        // Check kernel neighborhood
        for (let ky = -halfH; ky <= halfH; ky++) {
          for (let kx = -halfW; kx <= halfW; kx++) {
            const ny = y + ky;
            const nx = x + kx;

            if (ny >= 0 && ny < current.height && nx >= 0 && nx < current.width) {
              const idx = ny * current.width + nx;
              if (current.data[idx] > maxVal) {
                maxVal = current.data[idx];
              }
            }
          }
        }

        result[y * current.width + x] = maxVal;
        pixelCount++;

        // Yield periodically
        if (pixelCount % YIELD_INTERVAL === 0) {
          await yieldToMain();
        }
      }
    }

    current = { data: result, width: current.width, height: current.height };
  }

  return current;
}

/**
 * Erode binary image with rectangular kernel (async with yielding).
 * A pixel becomes white only if ALL pixels in the kernel neighborhood are white.
 */
async function erode(binary: GrayImage, kernelWidth: number, kernelHeight: number): Promise<GrayImage> {
  const result = new Uint8Array(binary.data.length);
  const halfW = Math.floor(kernelWidth / 2);
  const halfH = Math.floor(kernelHeight / 2);
  let pixelCount = 0;

  for (let y = 0; y < binary.height; y++) {
    for (let x = 0; x < binary.width; x++) {
      let minVal = 255;

      // Check kernel neighborhood
      for (let ky = -halfH; ky <= halfH; ky++) {
        for (let kx = -halfW; kx <= halfW; kx++) {
          const ny = y + ky;
          const nx = x + kx;

          if (ny >= 0 && ny < binary.height && nx >= 0 && nx < binary.width) {
            const idx = ny * binary.width + nx;
            if (binary.data[idx] < minVal) {
              minVal = binary.data[idx];
            }
          } else {
            minVal = 0;
          }
        }
      }

      result[y * binary.width + x] = minVal;
      pixelCount++;

      if (pixelCount % YIELD_INTERVAL === 0) {
        await yieldToMain();
      }
    }
  }

  return { data: result, width: binary.width, height: binary.height };
}

/**
 * Morphological opening: erosion followed by dilation (async).
 * Removes small white regions while preserving larger structures.
 */
async function morphOpen(binary: GrayImage, kernelWidth: number, kernelHeight: number): Promise<GrayImage> {
  const eroded = await erode(binary, kernelWidth, kernelHeight);
  return dilate(eroded, kernelWidth, kernelHeight, 1);
}

/**
 * Find connected components in binary image and return their bounding boxes (async).
 * Uses flood-fill algorithm with 4-connectivity.
 */
async function findContourBoundingBoxes(binary: GrayImage): Promise<ContourBox[]> {
  const { data, width, height } = binary;
  const visited = new Uint8Array(data.length);
  const boxes: ContourBox[] = [];
  let totalPixelsProcessed = 0;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x;

      // Skip if not white or already visited
      if (data[idx] === 0 || visited[idx]) continue;

      // Flood fill to find connected component
      let minX = x,
        maxX = x,
        minY = y,
        maxY = y;
      const stack: [number, number][] = [[x, y]];
      visited[idx] = 1;
      let fillCount = 0;

      while (stack.length > 0) {
        const [cx, cy] = stack.pop()!;
        fillCount++;

        // Update bounding box
        if (cx < minX) minX = cx;
        if (cx > maxX) maxX = cx;
        if (cy < minY) minY = cy;
        if (cy > maxY) maxY = cy;

        // Check 4-connected neighbors
        const neighbors: [number, number][] = [
          [cx - 1, cy],
          [cx + 1, cy],
          [cx, cy - 1],
          [cx, cy + 1],
        ];

        for (const [nx, ny] of neighbors) {
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const nidx = ny * width + nx;
            if (data[nidx] > 0 && !visited[nidx]) {
              visited[nidx] = 1;
              stack.push([nx, ny]);
            }
          }
        }

        // Yield during large flood fills
        if (fillCount % 10000 === 0) {
          await yieldToMain();
        }
      }

      boxes.push({
        x: minX,
        y: minY,
        w: maxX - minX + 1,
        h: maxY - minY + 1,
      });

      totalPixelsProcessed += fillCount;
      if (totalPixelsProcessed % YIELD_INTERVAL === 0) {
        await yieldToMain();
      }
    }
  }

  return boxes;
}

// ============================================================================
// BBox Overlap Detection
// ============================================================================

/**
 * Check if a bounding box overlaps with any existing boxes.
 * Uses IoU and center-inside checks.
 */
export function bboxOverlapsAny(
  bbox: BBox,
  existingBboxes: BBox[],
  overlapThreshold: number = THRESHOLDS.overlapIoU
): boolean {
  const newCx = bboxCenterX(bbox);
  const newCy = bboxCenterY(bbox);

  for (const existing of existingBboxes) {
    // Check IoU
    if (bboxIoU(bbox, existing) > overlapThreshold) {
      return true;
    }

    // Check if new box center is inside existing box
    if (
      existing.x0 <= newCx &&
      newCx <= existing.x1 &&
      existing.y0 <= newCy &&
      newCy <= existing.y1
    ) {
      return true;
    }
  }

  return false;
}

// ============================================================================
// Detection Functions
// ============================================================================

/**
 * Get ImageData from a canvas or image element.
 */
function getImageData(source: HTMLCanvasElement | ImageData): ImageData {
  if (source instanceof ImageData) {
    return source;
  }
  const ctx = source.getContext('2d');
  if (!ctx) throw new Error('Could not get 2D context from canvas');
  return ctx.getImageData(0, 0, source.width, source.height);
}

/**
 * Detect logo/image regions that OCR may have missed (async).
 * Targets larger, boxy regions that are not text-like.
 */
export async function detectLogoRegions(
  imageSource: HTMLCanvasElement | ImageData,
  pageNum: number,
  pageWidth: number,
  pageHeight: number,
  existingBboxes: BBox[] = [],
  minAreaPts: number = THRESHOLDS.logoMinAreaPts,
  minDimensionPts: number = THRESHOLDS.logoMinDimensionPts
): Promise<Word[]> {
  const imageData = getImageData(imageSource);
  const imgWidth = imageData.width;
  const imgHeight = imageData.height;
  const scaleX = pageWidth / imgWidth;
  const scaleY = pageHeight / imgHeight;

  // Convert to grayscale
  const gray = await toGrayscale(imageData);

  // Binary threshold (inverse) - this is fast, no need for async
  const binary = binaryThresholdInv(gray, THRESHOLDS.binaryThreshold);

  // Dilate to merge nearby pixels into regions
  const dilated = await dilate(binary, THRESHOLDS.dilateKernelSize, THRESHOLDS.dilateKernelSize, THRESHOLDS.dilateIterations);

  // Find contour bounding boxes
  const contours = await findContourBoundingBoxes(dilated);

  const words: Word[] = [];
  const existing = [...existingBboxes];

  for (const contour of contours) {
    const { x, y, w, h } = contour;

    // Convert to PDF coordinates
    const pdfX0 = x * scaleX;
    const pdfY0 = y * scaleY;
    const pdfX1 = (x + w) * scaleX;
    const pdfY1 = (y + h) * scaleY;
    const pdfW = pdfX1 - pdfX0;
    const pdfH = pdfY1 - pdfY0;
    const pdfArea = pdfW * pdfH;

    // Filter: must be large enough
    if (pdfArea < minAreaPts) continue;

    // Filter: must be reasonably sized (not text-like)
    if (pdfW < minDimensionPts && pdfH < minDimensionPts) continue;

    // Filter: aspect ratio should be boxy (not extreme like lines)
    const aspect = Math.max(pdfW, pdfH) / Math.max(Math.min(pdfW, pdfH), 0.1);
    if (aspect > THRESHOLDS.logoMaxAspectRatio) continue;

    const newBbox: BBox = { x0: pdfX0, y0: pdfY0, x1: pdfX1, y1: pdfY1 };

    if (!bboxOverlapsAny(newBbox, existing)) {
      words.push({
        wordId: -1,
        text: '',
        bbox: newBbox,
        page: pageNum,
        engine: 'pixel_detector',
        confidence: 0.0,
      });
      existing.push(newBbox);
    }
  }

  return words;
}

/**
 * Detect horizontal lines (separators, underlines, etc) - async.
 */
export async function detectHorizontalLines(
  imageSource: HTMLCanvasElement | ImageData,
  pageNum: number,
  pageWidth: number,
  pageHeight: number,
  existingBboxes: BBox[] = [],
  minWidthPts: number = THRESHOLDS.hLineMinWidthPts,
  maxHeightPts: number = THRESHOLDS.hLineMaxHeightPts
): Promise<Word[]> {
  const imageData = getImageData(imageSource);
  const imgWidth = imageData.width;
  const imgHeight = imageData.height;
  const scaleX = pageWidth / imgWidth;
  const scaleY = pageHeight / imgHeight;

  const minWidthPx = Math.floor(minWidthPts / scaleX);

  // Convert to grayscale
  const gray = await toGrayscale(imageData);

  // Binary threshold (inverse)
  const binary = binaryThresholdInv(gray, THRESHOLDS.binaryThreshold);

  // Morphological opening with horizontal kernel to detect horizontal lines
  const kernelWidth = Math.max(Math.floor(minWidthPx / 4), 3);
  const horizontal = await morphOpen(binary, kernelWidth, 1);

  // Find contour bounding boxes
  const contours = await findContourBoundingBoxes(horizontal);

  const words: Word[] = [];
  const existing = [...existingBboxes];

  for (const contour of contours) {
    const { x, y, w, h } = contour;

    const pdfX0 = x * scaleX;
    const pdfY0 = y * scaleY;
    const pdfX1 = (x + w) * scaleX;
    const pdfY1 = (y + h) * scaleY;
    const pdfW = pdfX1 - pdfX0;
    const pdfH = pdfY1 - pdfY0;

    // Must be wide and thin
    if (pdfW < minWidthPts || pdfH > maxHeightPts) continue;

    // Aspect ratio check (very horizontal)
    if (pdfW / Math.max(pdfH, 0.1) < THRESHOLDS.hLineMinAspectRatio) continue;

    const newBbox: BBox = { x0: pdfX0, y0: pdfY0, x1: pdfX1, y1: pdfY1 };

    if (!bboxOverlapsAny(newBbox, existing)) {
      words.push({
        wordId: -1,
        text: '',
        bbox: newBbox,
        page: pageNum,
        engine: 'pixel_detector',
        confidence: 0.0,
      });
      existing.push(newBbox);
    }
  }

  return words;
}

/**
 * Detect vertical lines (margins, separators, etc) - async.
 */
export async function detectVerticalLines(
  imageSource: HTMLCanvasElement | ImageData,
  pageNum: number,
  pageWidth: number,
  pageHeight: number,
  existingBboxes: BBox[] = [],
  minHeightPts: number = THRESHOLDS.vLineMinHeightPts,
  maxWidthPts: number = THRESHOLDS.vLineMaxWidthPts
): Promise<Word[]> {
  const imageData = getImageData(imageSource);
  const imgWidth = imageData.width;
  const imgHeight = imageData.height;
  const scaleX = pageWidth / imgWidth;
  const scaleY = pageHeight / imgHeight;

  const minHeightPx = Math.floor(minHeightPts / scaleY);

  // Convert to grayscale
  const gray = await toGrayscale(imageData);

  // Binary threshold (inverse)
  const binary = binaryThresholdInv(gray, THRESHOLDS.binaryThreshold);

  // Morphological opening with vertical kernel to detect vertical lines
  const kernelHeight = Math.max(Math.floor(minHeightPx / 4), 3);
  const vertical = await morphOpen(binary, 1, kernelHeight);

  // Find contour bounding boxes
  const contours = await findContourBoundingBoxes(vertical);

  const words: Word[] = [];
  const existing = [...existingBboxes];

  for (const contour of contours) {
    const { x, y, w, h } = contour;

    const pdfX0 = x * scaleX;
    const pdfY0 = y * scaleY;
    const pdfX1 = (x + w) * scaleX;
    const pdfY1 = (y + h) * scaleY;
    const pdfW = pdfX1 - pdfX0;
    const pdfH = pdfY1 - pdfY0;

    // Must be tall and thin
    if (pdfH < minHeightPts || pdfW > maxWidthPts) continue;

    // Aspect ratio check (very vertical)
    if (pdfH / Math.max(pdfW, 0.1) < THRESHOLDS.vLineMinAspectRatio) continue;

    const newBbox: BBox = { x0: pdfX0, y0: pdfY0, x1: pdfX1, y1: pdfY1 };

    if (!bboxOverlapsAny(newBbox, existing)) {
      words.push({
        wordId: -1,
        text: '',
        bbox: newBbox,
        page: pageNum,
        engine: 'pixel_detector',
        confidence: 0.0,
      });
      existing.push(newBbox);
    }
  }

  return words;
}

/**
 * Detect text-like regions that OCR may have missed (async).
 * Uses smaller thresholds than logo detection to catch small text.
 */
export async function detectTextRegions(
  imageSource: HTMLCanvasElement | ImageData,
  pageNum: number,
  pageWidth: number,
  pageHeight: number,
  existingBboxes: BBox[] = []
): Promise<Word[]> {
  const imageData = getImageData(imageSource);
  const imgWidth = imageData.width;
  const imgHeight = imageData.height;
  const scaleX = pageWidth / imgWidth;
  const scaleY = pageHeight / imgHeight;

  // Convert to grayscale
  const gray = await toGrayscale(imageData);

  // Binary threshold (inverse) - text becomes white on black
  const binary = binaryThresholdInv(gray, THRESHOLDS.binaryThreshold);

  // Dilate with smaller kernel to merge text characters without merging separate words too much
  const dilated = await dilate(
    binary,
    THRESHOLDS.textDilateKernelSize,
    THRESHOLDS.textDilateKernelSize,
    THRESHOLDS.textDilateIterations
  );

  // Find contour bounding boxes
  const contours = await findContourBoundingBoxes(dilated);

  const words: Word[] = [];
  const existing = [...existingBboxes];

  for (const contour of contours) {
    const { x, y, w, h } = contour;

    // Convert to page coordinates (using pixel coords since that's what we're working with)
    const pdfX0 = x * scaleX;
    const pdfY0 = y * scaleY;
    const pdfX1 = (x + w) * scaleX;
    const pdfY1 = (y + h) * scaleY;
    const pdfW = pdfX1 - pdfX0;
    const pdfH = pdfY1 - pdfY0;
    const pdfArea = pdfW * pdfH;

    // Filter: must meet minimum size
    if (pdfArea < THRESHOLDS.textMinAreaPts) continue;
    if (pdfW < THRESHOLDS.textMinWidthPts) continue;
    if (pdfH < THRESHOLDS.textMinHeightPts) continue;

    // Filter: text shouldn't be too tall (that's likely a logo)
    if (pdfH > THRESHOLDS.textMaxHeightPts) continue;

    // Filter: aspect ratio - text regions can be wide but not extremely so
    const aspect = Math.max(pdfW, pdfH) / Math.max(Math.min(pdfW, pdfH), 0.1);
    if (aspect > THRESHOLDS.textMaxAspectRatio) continue;

    const newBbox: BBox = { x0: pdfX0, y0: pdfY0, x1: pdfX1, y1: pdfY1 };

    if (!bboxOverlapsAny(newBbox, existing)) {
      words.push({
        wordId: -1,
        text: '',
        bbox: newBbox,
        page: pageNum,
        engine: 'pixel_detector',
        confidence: 0.0,
      });
      existing.push(newBbox);
    }
  }

  console.log(`[Pixel] Text region detection found ${words.length} regions`);
  return words;
}

/**
 * Main function to detect content missed by OCR (async).
 *
 * Targets:
 * 1. Text-like regions (small text that OCR missed)
 * 2. Horizontal lines (separators, underlines)
 * 3. Vertical lines (margins, borders)
 * 4. Logo/image regions (large, boxy)
 */
export async function detectMissedContent(
  imageSource: HTMLCanvasElement | ImageData,
  pageNum: number,
  pageWidth: number,
  pageHeight: number,
  existingBboxes: BBox[] = []
): Promise<Word[]> {
  const allDetected: Word[] = [];
  const existing = [...existingBboxes];

  // Detect text regions first (highest priority - catches OCR misses)
  const textRegions = await detectTextRegions(imageSource, pageNum, pageWidth, pageHeight, existing);
  allDetected.push(...textRegions);
  existing.push(...textRegions.map((w) => w.bbox));

  // Detect horizontal lines
  const hLines = await detectHorizontalLines(imageSource, pageNum, pageWidth, pageHeight, existing);
  allDetected.push(...hLines);
  existing.push(...hLines.map((w) => w.bbox));

  // Detect vertical lines
  const vLines = await detectVerticalLines(imageSource, pageNum, pageWidth, pageHeight, existing);
  allDetected.push(...vLines);
  existing.push(...vLines.map((w) => w.bbox));

  // Detect logo regions
  const logos = await detectLogoRegions(imageSource, pageNum, pageWidth, pageHeight, existing);
  allDetected.push(...logos);

  return allDetected;
}

// ============================================================================
// Sanity Check / Test Helper
// ============================================================================

/**
 * Run sanity check to verify detection is working (async).
 * Tests that a known region can be detected.
 */
export async function runSanityCheck(
  imageSource: HTMLCanvasElement | ImageData,
  pageWidth: number,
  pageHeight: number,
  expectedBbox: BBox,
  tolerance: number = 20
): Promise<{ passed: boolean; detected: Word[]; closest: Word | null; distance: number }> {
  const detected = await detectMissedContent(imageSource, 0, pageWidth, pageHeight, []);

  // Find closest detected region to expected bbox
  let closest: Word | null = null;
  let minDistance = Infinity;

  const expectedCx = (expectedBbox.x0 + expectedBbox.x1) / 2;
  const expectedCy = (expectedBbox.y0 + expectedBbox.y1) / 2;

  for (const word of detected) {
    const cx = (word.bbox.x0 + word.bbox.x1) / 2;
    const cy = (word.bbox.y0 + word.bbox.y1) / 2;
    const dist = Math.sqrt((cx - expectedCx) ** 2 + (cy - expectedCy) ** 2);

    if (dist < minDistance) {
      minDistance = dist;
      closest = word;
    }
  }

  // Check if closest is within tolerance
  const passed = closest !== null && minDistance <= tolerance;

  return { passed, detected, closest, distance: minDistance };
}

/**
 * Expected bbox for sanity check on peter_lou_50dpi.pdf page 1.
 * This is the logo/image region at the top of the page.
 * Note: Coordinates are in canvas pixels, not PDF points.
 * The paw logo is detected at approximately (102, 117) - (189, 220).
 */
export const SANITY_CHECK_EXPECTED_BBOX: BBox = {
  x0: 102,
  y0: 117,
  x1: 189,
  y1: 220,
};
