/**
 * Browser OCR Validation Script
 *
 * Compares browser-extracted words against ground truth CSV.
 * Run with: npx ts-node validate.ts <extracted.csv> <ground_truth.csv>
 *
 * Or import and use validateExtraction() directly.
 */

import * as fs from 'fs';

interface Word {
  page: number;
  text: string;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

interface ValidationResult {
  totalGroundTruth: number;
  totalExtracted: number;
  matched: number;
  precision: number;
  recall: number;
  f1: number;
  unmatchedGt: Word[];
  unmatchedExtracted: Word[];
}

/**
 * Parse CSV file into Word array
 */
function parseCsv(filepath: string): Word[] {
  const content = fs.readFileSync(filepath, 'utf-8');
  const lines = content.trim().split('\n');
  const header = lines[0].toLowerCase();

  // Determine column indices
  const cols = header.split(',').map(c => c.trim());
  const pageIdx = cols.findIndex(c => c === 'page');
  const textIdx = cols.findIndex(c => c === 'text');
  const x0Idx = cols.findIndex(c => c === 'x0');
  const y0Idx = cols.findIndex(c => c === 'y0');
  const x1Idx = cols.findIndex(c => c === 'x1');
  const y1Idx = cols.findIndex(c => c === 'y1');

  const words: Word[] = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;

    // Handle quoted fields (text might contain commas)
    const values = parseCSVLine(line);

    // Skip pixel_detector entries and empty text
    const text = values[textIdx]?.replace(/^"|"$/g, '').trim();
    if (!text) continue;

    // Check if this is a pixel_detector entry (engine column if present)
    const engineIdx = cols.findIndex(c => c === 'engine');
    if (engineIdx >= 0 && values[engineIdx] === 'pixel_detector') continue;

    words.push({
      page: parseInt(values[pageIdx]) || 0,
      text,
      x0: parseFloat(values[x0Idx]) || 0,
      y0: parseFloat(values[y0Idx]) || 0,
      x1: parseFloat(values[x1Idx]) || 0,
      y1: parseFloat(values[y1Idx]) || 0,
    });
  }

  return words;
}

/**
 * Parse a single CSV line handling quoted fields
 */
function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current);
      current = '';
    } else {
      current += char;
    }
  }
  result.push(current);

  return result;
}

/**
 * Calculate IoU (Intersection over Union) between two bboxes
 */
function calculateIoU(a: Word, b: Word): number {
  const x0 = Math.max(a.x0, b.x0);
  const y0 = Math.max(a.y0, b.y0);
  const x1 = Math.min(a.x1, b.x1);
  const y1 = Math.min(a.y1, b.y1);

  if (x1 <= x0 || y1 <= y0) return 0;

  const intersection = (x1 - x0) * (y1 - y0);
  const areaA = (a.x1 - a.x0) * (a.y1 - a.y0);
  const areaB = (b.x1 - b.x0) * (b.y1 - b.y0);
  const union = areaA + areaB - intersection;

  return union > 0 ? intersection / union : 0;
}

/**
 * Normalize text for comparison
 */
function normalizeText(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s]/g, '') // Remove punctuation
    .trim();
}

/**
 * Check if two words match (text similarity + bbox overlap)
 */
function wordsMatch(gt: Word, extracted: Word, iouThreshold = 0.3): boolean {
  // Must be same page
  if (gt.page !== extracted.page) return false;

  // Check bbox overlap
  const iou = calculateIoU(gt, extracted);
  if (iou < iouThreshold) return false;

  // Check text similarity
  const gtNorm = normalizeText(gt.text);
  const extNorm = normalizeText(extracted.text);

  // Exact match after normalization
  if (gtNorm === extNorm) return true;

  // One contains the other (for partial matches)
  if (gtNorm.includes(extNorm) || extNorm.includes(gtNorm)) return true;

  // Levenshtein distance for fuzzy matching
  const maxLen = Math.max(gtNorm.length, extNorm.length);
  if (maxLen === 0) return false;

  const distance = levenshtein(gtNorm, extNorm);
  const similarity = 1 - distance / maxLen;

  return similarity >= 0.7; // 70% similarity threshold
}

/**
 * Levenshtein distance between two strings
 */
function levenshtein(a: string, b: string): number {
  const matrix: number[][] = [];

  for (let i = 0; i <= b.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= a.length; j++) {
    matrix[0][j] = j;
  }

  for (let i = 1; i <= b.length; i++) {
    for (let j = 1; j <= a.length; j++) {
      if (b[i - 1] === a[j - 1]) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1,
          matrix[i][j - 1] + 1,
          matrix[i - 1][j] + 1
        );
      }
    }
  }

  return matrix[b.length][a.length];
}

/**
 * Scale word coordinates by a factor
 */
function scaleWord(word: Word, scale: number): Word {
  return {
    ...word,
    x0: word.x0 / scale,
    y0: word.y0 / scale,
    x1: word.x1 / scale,
    y1: word.y1 / scale,
  };
}

/**
 * Validate extracted words against ground truth
 * @param scale - Scale factor to apply to extracted coordinates (e.g., 1.5 for 150 DPI rendering)
 */
export function validateExtraction(
  extractedWords: Word[],
  groundTruthWords: Word[],
  iouThreshold = 0.3,
  scale = 1.0
): ValidationResult {
  // Scale extracted coordinates if needed
  const scaledExtracted = scale !== 1.0
    ? extractedWords.map(w => scaleWord(w, scale))
    : extractedWords;
  const matchedGt = new Set<number>();
  const matchedExt = new Set<number>();

  // Try to match each ground truth word
  for (let gi = 0; gi < groundTruthWords.length; gi++) {
    const gt = groundTruthWords[gi];

    for (let ei = 0; ei < scaledExtracted.length; ei++) {
      if (matchedExt.has(ei)) continue;

      const ext = scaledExtracted[ei];
      if (wordsMatch(gt, ext, iouThreshold)) {
        matchedGt.add(gi);
        matchedExt.add(ei);
        break;
      }
    }
  }

  const matched = matchedGt.size;
  const totalGt = groundTruthWords.length;
  const totalExt = scaledExtracted.length;

  const precision = totalExt > 0 ? matched / totalExt : 0;
  const recall = totalGt > 0 ? matched / totalGt : 0;
  const f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

  // Collect unmatched words (use scaled coordinates for consistency)
  const unmatchedGt = groundTruthWords.filter((_, i) => !matchedGt.has(i));
  const unmatchedExtracted = scaledExtracted.filter((_, i) => !matchedExt.has(i));

  return {
    totalGroundTruth: totalGt,
    totalExtracted: totalExt,
    matched,
    precision,
    recall,
    f1,
    unmatchedGt,
    unmatchedExtracted,
  };
}

/**
 * Main CLI entry point
 */
function main() {
  const args = process.argv.slice(2);

  if (args.length < 2) {
    console.log('Usage: npx tsx validate.ts <extracted.csv> <ground_truth.csv> [options]');
    console.log('');
    console.log('Compares browser-extracted words against ground truth.');
    console.log('');
    console.log('Options:');
    console.log('  --scale N    Scale factor for extracted coordinates (default: 1.5 for browser)');
    console.log('  --verbose    Show unmatched words');
    console.log('');
    console.log('Example:');
    console.log('  npx tsx validate.ts portadoc-export.csv ground_truth.csv --scale 1.5');
    process.exit(1);
  }

  const extractedPath = args[0];
  const gtPath = args[1];

  // Parse scale option
  let scale = 1.5; // Default for browser rendering at 150 DPI
  const scaleIdx = args.indexOf('--scale');
  if (scaleIdx >= 0 && args[scaleIdx + 1]) {
    scale = parseFloat(args[scaleIdx + 1]);
  }

  if (!fs.existsSync(extractedPath)) {
    console.error(`Error: Extracted file not found: ${extractedPath}`);
    process.exit(1);
  }

  if (!fs.existsSync(gtPath)) {
    console.error(`Error: Ground truth file not found: ${gtPath}`);
    process.exit(1);
  }

  console.log('Loading files...');
  const extracted = parseCsv(extractedPath);
  const gt = parseCsv(gtPath);

  console.log(`Extracted: ${extracted.length} words`);
  console.log(`Ground truth: ${gt.length} words`);
  console.log('');

  console.log(`Scale factor: ${scale}`);
  console.log('Validating...');
  const result = validateExtraction(extracted, gt, 0.3, scale);

  console.log('');
  console.log('=== VALIDATION RESULTS ===');
  console.log(`Ground Truth Words: ${result.totalGroundTruth}`);
  console.log(`Extracted Words:    ${result.totalExtracted}`);
  console.log(`Matched:            ${result.matched}`);
  console.log('');
  console.log(`Precision: ${(result.precision * 100).toFixed(1)}%`);
  console.log(`Recall:    ${(result.recall * 100).toFixed(1)}%`);
  console.log(`F1 Score:  ${(result.f1 * 100).toFixed(1)}%`);
  console.log('');

  if (result.unmatchedGt.length > 0 && args.includes('--verbose')) {
    console.log('--- Unmatched Ground Truth (first 20) ---');
    result.unmatchedGt.slice(0, 20).forEach(w => {
      console.log(`  Page ${w.page}: "${w.text}" at (${w.x0.toFixed(0)},${w.y0.toFixed(0)})`);
    });
    console.log('');
  }

  if (result.unmatchedExtracted.length > 0 && args.includes('--verbose')) {
    console.log('--- Unmatched Extracted (first 20) ---');
    result.unmatchedExtracted.slice(0, 20).forEach(w => {
      console.log(`  Page ${w.page}: "${w.text}" at (${w.x0.toFixed(0)},${w.y0.toFixed(0)})`);
    });
  }

  // Exit with non-zero if F1 < 80%
  if (result.f1 < 0.8) {
    console.log('');
    console.log(`WARNING: F1 score ${(result.f1 * 100).toFixed(1)}% is below 80% target`);
    process.exit(1);
  }
}

// Run if called directly
const isMain = import.meta.url === `file://${process.argv[1]}` ||
               process.argv[1]?.endsWith('validate.ts');
if (isMain) {
  main();
}
