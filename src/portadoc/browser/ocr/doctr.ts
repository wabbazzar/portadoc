/**
 * docTR-TFJS wrapper for browser-based OCR.
 * Adapted from https://github.com/mindee/doctr-tfjs-demo
 */

import * as tf from '@tensorflow/tfjs';
import { Word, BBox } from '../models';
import { adjustConfidence } from '../config';
import { resolveAssetPath } from '../basePath';

// Model paths - resolved at runtime with base URL for GitHub Pages
function getDetectionModelUrl(): string {
  return resolveAssetPath('models/db_mobilenet_v2/model.json');
}
function getRecognitionModelUrl(): string {
  return resolveAssetPath('models/crnn_mobilenet_v2/model.json');
}

// Normalization constants
const DET_MEAN = 0.785;
const DET_STD = 0.275;
const DET_SIZE = 512;

const REC_MEAN = 0.694;
const REC_STD = 0.298;
const REC_HEIGHT = 32;
const REC_WIDTH = 128;

// Character vocabulary for recognition (must match docTR training vocab - 126 chars)
// From https://github.com/mindee/doctr-tfjs-demo/blob/main/src/common/constants.ts
const VOCAB = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~°£€¥¢฿àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ";
const BLANK_INDEX = 126; // CTC blank token is at index 126 (after all vocab chars)

// Models
let detectionModel: tf.GraphModel | null = null;
let recognitionModel: tf.GraphModel | null = null;

/**
 * Load the detection and recognition models.
 */
export async function loadModels(
  onProgress?: (progress: number, status: string) => void
): Promise<void> {
  if (detectionModel && recognitionModel) {
    return; // Already loaded
  }

  onProgress?.(0, 'Loading detection model...');

  try {
    detectionModel = await tf.loadGraphModel(getDetectionModelUrl());
    onProgress?.(0.4, 'Detection model loaded');

    // Warm up detection model
    const warmupDet = tf.zeros([1, DET_SIZE, DET_SIZE, 3]);
    await (detectionModel.predict(warmupDet) as tf.Tensor).data();
    warmupDet.dispose();
    onProgress?.(0.5, 'Loading recognition model...');

    recognitionModel = await tf.loadGraphModel(getRecognitionModelUrl());
    onProgress?.(0.9, 'Recognition model loaded');

    // Warm up recognition model (use executeAsync for dynamic ops)
    const warmupRec = tf.zeros([1, REC_HEIGHT, REC_WIDTH, 3]);
    await (await recognitionModel.executeAsync(warmupRec) as tf.Tensor).data();
    warmupRec.dispose();

    onProgress?.(1.0, 'Models ready');
  } catch (error) {
    console.error('Failed to load docTR models:', error);
    throw new Error(`Failed to load docTR models: ${(error as Error).message}`);
  }
}

/**
 * Preprocess image for detection model.
 * Matches Mindee's normalization: (pixel - 255*mean) / (255*std)
 */
function preprocessForDetection(imageData: ImageData): tf.Tensor4D {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(imageData);

    // Resize to DET_SIZE x DET_SIZE
    const resized = tf.image.resizeNearestNeighbor(tensor, [DET_SIZE, DET_SIZE]);

    // Normalize: (pixel - 255*mean) / (255*std)
    const mean = tf.scalar(255 * DET_MEAN);
    const std = tf.scalar(255 * DET_STD);
    const normalized = resized.toFloat().sub(mean).div(std);

    // Add batch dimension
    return normalized.expandDims(0) as tf.Tensor4D;
  });
}

/**
 * Preprocess a crop for recognition model with aspect-ratio-preserving resize.
 * Matches Mindee's preprocessing with padding.
 */
function preprocessForRecognition(
  imageData: ImageData,
  bbox: { x0: number; y0: number; x1: number; y1: number }
): tf.Tensor4D {
  return tf.tidy(() => {
    const tensor = tf.browser.fromPixels(imageData);

    // Crop the region
    const [imgH, imgW] = [imageData.height, imageData.width];
    const y0 = Math.max(0, Math.floor(bbox.y0));
    const x0 = Math.max(0, Math.floor(bbox.x0));
    const y1 = Math.min(imgH, Math.ceil(bbox.y1));
    const x1 = Math.min(imgW, Math.ceil(bbox.x1));

    const cropH = y1 - y0;
    const cropW = x1 - x0;

    if (cropH <= 0 || cropW <= 0) {
      // Return zero tensor for invalid crops
      return tf.zeros([1, REC_HEIGHT, REC_WIDTH, 3]) as tf.Tensor4D;
    }

    const cropped = tf.slice(tensor, [y0, x0, 0], [cropH, cropW, 3]);

    // Aspect-ratio-preserving resize with padding (matching Mindee's approach)
    const aspectRatio = REC_WIDTH / REC_HEIGHT;
    let resizeTarget: [number, number];
    let paddings: [[number, number], [number, number], [number, number]];

    if (aspectRatio * cropH > cropW) {
      // Height-constrained: resize to REC_HEIGHT and pad width
      const newW = Math.round((REC_HEIGHT * cropW) / cropH);
      resizeTarget = [REC_HEIGHT, newW];
      paddings = [[0, 0], [0, REC_WIDTH - newW], [0, 0]];
    } else {
      // Width-constrained: resize to REC_WIDTH and pad height
      const newH = Math.round((REC_WIDTH * cropH) / cropW);
      resizeTarget = [newH, REC_WIDTH];
      paddings = [[0, REC_HEIGHT - newH], [0, 0], [0, 0]];
    }

    const resized = tf.image.resizeNearestNeighbor(cropped, resizeTarget);
    const padded = tf.pad(resized, paddings, 0);

    // Normalize: (pixel - 255*mean) / (255*std)
    const mean = tf.scalar(255 * REC_MEAN);
    const std = tf.scalar(255 * REC_STD);
    const normalized = padded.toFloat().sub(mean).div(std);

    return normalized.expandDims(0) as tf.Tensor4D;
  });
}

/**
 * Extract text regions from detection heatmap using thresholding and connected components.
 * Returns bounding boxes of detected text regions.
 */
function extractBoxesFromHeatmap(
  heatmap: Float32Array,
  originalWidth: number,
  originalHeight: number,
  threshold: number = 0.15
): Array<{ x0: number; y0: number; x1: number; y1: number }> {
  const boxes: Array<{ x0: number; y0: number; x1: number; y1: number }> = [];

  // Scale factors from detection size to original
  const scaleX = originalWidth / DET_SIZE;
  const scaleY = originalHeight / DET_SIZE;

  // Create binary mask
  const mask = new Uint8Array(DET_SIZE * DET_SIZE);
  for (let i = 0; i < heatmap.length; i++) {
    mask[i] = heatmap[i] > threshold ? 1 : 0;
  }

  // Simple morphological opening to remove noise (erode then dilate)
  const eroded = morphErode(mask, DET_SIZE, DET_SIZE, 2);
  const opened = morphDilate(eroded, DET_SIZE, DET_SIZE, 2);

  // Connected component labeling with union-find
  const labels = new Int32Array(DET_SIZE * DET_SIZE).fill(-1);
  let nextLabel = 0;

  for (let y = 0; y < DET_SIZE; y++) {
    for (let x = 0; x < DET_SIZE; x++) {
      const idx = y * DET_SIZE + x;
      if (opened[idx] === 0 || labels[idx] >= 0) continue;

      // BFS to label connected region
      const queue: Array<[number, number]> = [[x, y]];
      const label = nextLabel++;
      let minX = x, maxX = x, minY = y, maxY = y;

      while (queue.length > 0) {
        const [cx, cy] = queue.shift()!;
        const cIdx = cy * DET_SIZE + cx;

        if (cx < 0 || cx >= DET_SIZE || cy < 0 || cy >= DET_SIZE) continue;
        if (opened[cIdx] === 0 || labels[cIdx] >= 0) continue;

        labels[cIdx] = label;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);

        // 8-connectivity for better region detection
        queue.push(
          [cx - 1, cy], [cx + 1, cy], [cx, cy - 1], [cx, cy + 1],
          [cx - 1, cy - 1], [cx + 1, cy - 1], [cx - 1, cy + 1], [cx + 1, cy + 1]
        );
      }

      // Filter by size - keep reasonable regions
      const width = maxX - minX + 1;
      const height = maxY - minY + 1;

      // Minimum size (larger than noise) and maximum size (not full page)
      if (width >= 4 && height >= 3 && width < DET_SIZE * 0.8 && height < DET_SIZE * 0.3) {
        // Add padding for better recognition
        const pad = Math.max(2, Math.min(width, height) * 0.1);
        boxes.push({
          x0: Math.max(0, (minX - pad)) * scaleX,
          y0: Math.max(0, (minY - pad)) * scaleY,
          x1: Math.min(DET_SIZE, (maxX + 1 + pad)) * scaleX,
          y1: Math.min(DET_SIZE, (maxY + 1 + pad)) * scaleY,
        });
      }
    }
  }

  console.log(`Detection found ${boxes.length} text regions`);
  return boxes;
}

/**
 * Simple erosion operation
 */
function morphErode(input: Uint8Array, width: number, height: number, kernelSize: number): Uint8Array {
  const output = new Uint8Array(width * height);
  const half = Math.floor(kernelSize / 2);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let allOnes = true;
      for (let ky = -half; ky <= half && allOnes; ky++) {
        for (let kx = -half; kx <= half && allOnes; kx++) {
          const ny = y + ky;
          const nx = x + kx;
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            if (input[ny * width + nx] === 0) allOnes = false;
          }
        }
      }
      output[y * width + x] = allOnes ? 1 : 0;
    }
  }
  return output;
}

/**
 * Simple dilation operation
 */
function morphDilate(input: Uint8Array, width: number, height: number, kernelSize: number): Uint8Array {
  const output = new Uint8Array(width * height);
  const half = Math.floor(kernelSize / 2);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let anyOne = false;
      for (let ky = -half; ky <= half && !anyOne; ky++) {
        for (let kx = -half; kx <= half && !anyOne; kx++) {
          const ny = y + ky;
          const nx = x + kx;
          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            if (input[ny * width + nx] === 1) anyOne = true;
          }
        }
      }
      output[y * width + x] = anyOne ? 1 : 0;
    }
  }
  return output;
}

interface RecognitionResult {
  text: string;
  confidence: number;  // 0-100 percentage
}

/**
 * Decode recognition output to text using CTC decoding.
 * Standard CTC: collapse consecutive identical characters, remove blanks.
 * Returns text and confidence based on softmax probabilities.
 */
function decodeRecognition(predictionTensor: tf.Tensor): RecognitionResult {
  // Apply softmax to get probabilities along last axis
  const probabilities = tf.softmax(predictionTensor, -1);

  // Get argmax for each timestep along last axis
  const bestPath = tf.argMax(probabilities, -1);

  // Squeeze batch dimension if present to get [sequence_length]
  const squeezedPath = tf.squeeze(bestPath);
  const squeezedProbs = tf.squeeze(probabilities);

  // Get the data
  const indices = squeezedPath.dataSync();
  const probsData = squeezedProbs.dataSync();

  // Get shape for indexing into probabilities
  const numClasses = VOCAB.length + 1; // vocab + blank

  // CTC decode: collapse consecutive identical chars, remove blanks
  // Also collect confidence scores for decoded characters
  let text = '';
  let prevIdx = -1;
  const charConfidences: number[] = [];

  for (let t = 0; t < indices.length; t++) {
    const k = indices[t];
    // Skip blanks and consecutive identical characters
    if (k !== BLANK_INDEX && k !== prevIdx) {
      if (k < VOCAB.length) {
        text += VOCAB[k];
        // Get the probability for this character at this timestep
        const prob = probsData[t * numClasses + k];
        charConfidences.push(prob);
      }
    }
    prevIdx = k;
  }

  // Dispose tensors
  probabilities.dispose();
  bestPath.dispose();
  squeezedPath.dispose();
  squeezedProbs.dispose();

  // Compute overall confidence as geometric mean of character probabilities
  // This penalizes low-confidence characters appropriately
  let confidence = 0;
  if (charConfidences.length > 0) {
    // Geometric mean: (p1 * p2 * ... * pn) ^ (1/n)
    // Compute in log space to avoid underflow: exp(mean(log(pi)))
    const logSum = charConfidences.reduce((sum, p) => sum + Math.log(Math.max(p, 1e-10)), 0);
    const geometricMean = Math.exp(logSum / charConfidences.length);
    confidence = geometricMean * 100; // Convert to percentage
  }

  return { text, confidence };
}

/**
 * Split a text region into individual words based on horizontal gaps.
 */
function splitIntoWords(
  text: string,
  bbox: { x0: number; y0: number; x1: number; y1: number }
): Array<{ text: string; bbox: { x0: number; y0: number; x1: number; y1: number } }> {
  const words = text.trim().split(/\s+/).filter(w => w.length > 0);

  if (words.length === 0) return [];
  if (words.length === 1) {
    return [{ text: words[0], bbox }];
  }

  // Estimate character width and distribute words
  const totalChars = words.reduce((sum, w) => sum + w.length, 0) + words.length - 1;
  const charWidth = (bbox.x1 - bbox.x0) / totalChars;

  const result: Array<{ text: string; bbox: { x0: number; y0: number; x1: number; y1: number } }> = [];
  let currentX = bbox.x0;

  for (const word of words) {
    const wordWidth = word.length * charWidth;
    result.push({
      text: word,
      bbox: {
        x0: currentX,
        y0: bbox.y0,
        x1: currentX + wordWidth,
        y1: bbox.y1,
      },
    });
    currentX += wordWidth + charWidth; // +charWidth for space
  }

  return result;
}

/**
 * Extract words from an image using docTR-TFJS.
 */
export async function extractWithDoctr(
  imageData: ImageData,
  pageIndex: number,
  onProgress?: (progress: number) => void
): Promise<Word[]> {
  if (!detectionModel || !recognitionModel) {
    await loadModels((p) => onProgress?.(p * 0.3));
  }

  onProgress?.(0.3);

  // Run detection
  const detInput = preprocessForDetection(imageData);
  const detOutput = (await detectionModel!.predict(detInput)) as tf.Tensor;
  const heatmap = await detOutput.data() as Float32Array;

  detInput.dispose();
  detOutput.dispose();

  onProgress?.(0.5);

  // Extract boxes from heatmap
  const boxes = extractBoxesFromHeatmap(heatmap, imageData.width, imageData.height);

  if (boxes.length === 0) {
    return [];
  }

  // Sort boxes by reading order (top-to-bottom, left-to-right)
  boxes.sort((a, b) => {
    const yDiff = a.y0 - b.y0;
    if (Math.abs(yDiff) > 10) return yDiff;
    return a.x0 - b.x0;
  });

  const words: Word[] = [];

  // Recognize each box
  for (let i = 0; i < boxes.length; i++) {
    const box = boxes[i];

    try {
      const recInput = preprocessForRecognition(imageData, box);
      // Use executeAsync for models with dynamic ops (bidirectional LSTM)
      const recOutput = (await recognitionModel!.executeAsync(recInput)) as tf.Tensor;

      // Decode the recognition output with confidence
      const result = decodeRecognition(recOutput);

      recInput.dispose();
      recOutput.dispose();

      if (result.text.trim()) {
        // Split into individual words
        const wordParts = splitIntoWords(result.text, box);

        for (const part of wordParts) {
          words.push({
            page: pageIndex,
            wordId: words.length,
            text: part.text,
            bbox: part.bbox as BBox,
            engine: 'doctr',
            confidence: adjustConfidence('doctr', result.confidence),
          });
        }
      }
    } catch (err) {
      console.warn('Recognition failed for box:', box, err);
    }

    onProgress?.(0.5 + (i / boxes.length) * 0.5);
  }

  return words;
}

/**
 * Check if models are loaded.
 */
export function isModelLoaded(): boolean {
  return detectionModel !== null && recognitionModel !== null;
}

/**
 * Dispose models to free memory.
 */
export function disposeModels(): void {
  if (detectionModel) {
    detectionModel.dispose();
    detectionModel = null;
  }
  if (recognitionModel) {
    recognitionModel.dispose();
    recognitionModel = null;
  }
}
