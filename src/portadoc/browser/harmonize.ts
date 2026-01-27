/**
 * Harmonization module for browser-based multi-engine OCR.
 * Simplified port of src/portadoc/harmonize.py for 2-engine scenario.
 */

import { Word, ExtractedWord, BBox, HarmonizeContribution, HarmonizeDetails, bboxIoU, bboxCenterX, bboxCenterY } from './models';

// Configuration defaults
const DEFAULT_IOU_THRESHOLD = 0.3;
const DEFAULT_TEXT_MATCH_BONUS = 0.15;
const DEFAULT_CENTER_DISTANCE_MAX = 12.0;
const DEFAULT_Y_BAND_TOLERANCE = 10.0;

interface MatchResult {
  text: string;
  confidence: number;
  engine: string;
  wordIndex: number;
  bbox: BBox;
  iouScore: number;
  matchType: 'exact_text_iou' | 'iou_only' | 'center_distance';
  textMatches: boolean;
}

/**
 * Find the best matching word from secondary results for a primary word.
 * Uses IoU-based matching with text-aware bonus.
 */
function findWordMatch(
  primary: Word,
  secondaryWords: Word[],
  matchedIndices: Set<number>,
  iouThreshold: number = DEFAULT_IOU_THRESHOLD,
  textMatchBonus: number = DEFAULT_TEXT_MATCH_BONUS,
  centerDistanceMax: number = DEFAULT_CENTER_DISTANCE_MAX,
  yBandTolerance: number = DEFAULT_Y_BAND_TOLERANCE
): MatchResult | null {
  let bestMatch: MatchResult | null = null;
  let bestScore = 0;

  const primaryTextLower = primary.text.toLowerCase().trim();
  const primaryCx = bboxCenterX(primary.bbox);
  const primaryCy = bboxCenterY(primary.bbox);

  for (let i = 0; i < secondaryWords.length; i++) {
    if (matchedIndices.has(i)) continue;

    const word = secondaryWords[i];
    if (word.page !== primary.page) continue;

    // Check vertical alignment
    const yDiff = Math.abs(bboxCenterY(word.bbox) - primaryCy);
    if (yDiff > yBandTolerance) continue;

    const iou = bboxIoU(primary.bbox, word.bbox);
    const wordTextLower = word.text.toLowerCase().trim();
    const textMatches = primaryTextLower === wordTextLower;

    // Calculate effective threshold
    const effectiveThreshold = textMatches
      ? iouThreshold - textMatchBonus
      : iouThreshold;

    let isMatch = iou >= effectiveThreshold;
    let matchType: 'exact_text_iou' | 'iou_only' | 'center_distance' =
      textMatches ? 'exact_text_iou' : 'iou_only';

    // Fallback: center distance + text match
    if (!isMatch && textMatches && centerDistanceMax > 0) {
      const centerDist = Math.sqrt(
        Math.pow(bboxCenterX(word.bbox) - primaryCx, 2) +
          Math.pow(bboxCenterY(word.bbox) - primaryCy, 2)
      );
      if (centerDist <= centerDistanceMax) {
        isMatch = true;
        matchType = 'center_distance';
      }
    }

    if (isMatch) {
      const score = iou + (textMatches ? 0.5 : 0);
      if (score > bestScore) {
        bestScore = score;
        bestMatch = {
          text: word.text,
          confidence: word.confidence,
          engine: word.engine,
          wordIndex: i,
          bbox: word.bbox,
          iouScore: iou,
          matchType,
          textMatches,
        };
      }
    }
  }

  return bestMatch;
}

interface TextPickResult {
  text: string;
  reason: string;
}

/**
 * Pick the best text from multiple engine readings.
 * Simple confidence-based selection with text agreement bonus.
 */
function pickBestText(
  votes: Array<{ engine: string; text: string; confidence: number }>
): TextPickResult {
  if (votes.length === 0) return { text: '', reason: 'No votes' };
  if (votes.length === 1) {
    return {
      text: votes[0].text,
      reason: `Only ${votes[0].engine} (${votes[0].confidence.toFixed(0)}%)`,
    };
  }

  // If texts match (case-insensitive), return the higher confidence one
  const normalized = votes.map((v) => ({
    ...v,
    norm: v.text.toLowerCase().trim(),
  }));

  // Group by normalized text
  const groups = new Map<string, typeof normalized>();
  for (const vote of normalized) {
    const existing = groups.get(vote.norm) || [];
    existing.push(vote);
    groups.set(vote.norm, existing);
  }

  // Find the group with highest combined confidence
  let bestText = votes[0].text;
  let bestScore = 0;
  let bestReason = '';

  for (const [, group] of groups) {
    // Score = sum of confidences + bonus for agreement
    const agreementBonus = group.length > 1 ? 50 : 0;
    const score = group.reduce((sum, v) => sum + v.confidence, 0) + agreementBonus;

    if (score > bestScore) {
      bestScore = score;
      // Return the version with highest confidence
      const sortedGroup = group.sort((a, b) => b.confidence - a.confidence);
      bestText = sortedGroup[0].text;

      // Build reason
      if (group.length > 1) {
        bestReason = 'Text agreement (both engines matched)';
      } else {
        const winner = sortedGroup[0];
        const loser = votes.find((v) => v.engine !== winner.engine);
        if (loser) {
          bestReason = `Higher confidence: ${winner.engine} (${winner.confidence.toFixed(0)}%) vs ${loser.engine} (${loser.confidence.toFixed(0)}%)`;
        } else {
          bestReason = `${winner.engine} (${winner.confidence.toFixed(0)}%)`;
        }
      }
    }
  }

  return { text: bestText, reason: bestReason };
}

/**
 * Harmonize results from multiple OCR engines.
 * Primary engine provides bboxes, text is voted on.
 */
export function harmonizeWords(
  tesseractWords: Word[],
  doctrWords: Word[],
  primaryEngine: 'tesseract' | 'doctr' = 'tesseract'
): ExtractedWord[] {
  console.log('[Harmonize] Input - Tesseract:', tesseractWords.length, 'docTR:', doctrWords.length);

  // Debug: Check for specific words
  const tesseractTexts = tesseractWords.map(w => w.text.toLowerCase());
  console.log('[Harmonize] Tesseract has Peter:', tesseractTexts.some(t => t.includes('peter')));
  console.log('[Harmonize] Tesseract has Lou:', tesseractTexts.some(t => t === 'lou'));
  console.log('[Harmonize] Tesseract has INK:', tesseractTexts.some(t => t.includes('ink')));

  const primaryWords = primaryEngine === 'tesseract' ? tesseractWords : doctrWords;
  const secondaryWords =
    primaryEngine === 'tesseract' ? doctrWords : tesseractWords;
  const secondaryEngine = primaryEngine === 'tesseract' ? 'doctr' : 'tesseract';

  const result: ExtractedWord[] = [];
  const matchedSecondary = new Set<number>();
  const usedBboxes: Array<{ x0: number; y0: number; x1: number; y1: number }> =
    [];

  // Phase 1: Process all primary words
  for (const primary of primaryWords) {
    const votes: Array<{ engine: string; text: string; confidence: number }> = [
      { engine: primaryEngine, text: primary.text, confidence: primary.confidence },
    ];

    // Build contributions list
    const contributions: HarmonizeContribution[] = [
      {
        engine: primaryEngine,
        text: primary.text,
        confidence: primary.confidence,
        // No bbox field since this is the primary (bbox authority)
      },
    ];

    // Find matching secondary word
    const match = findWordMatch(primary, secondaryWords, matchedSecondary);
    let source = primaryEngine === 'tesseract' ? 'T' : 'D';
    let matchType: HarmonizeDetails['matchType'] = 'unmatched';
    let iouScore: number | undefined;
    let textAgreed = false;

    if (match) {
      matchedSecondary.add(match.wordIndex);
      votes.push({
        engine: secondaryEngine,
        text: match.text,
        confidence: match.confidence,
      });
      source += secondaryEngine === 'tesseract' ? 'T' : 'D';

      // Add secondary contribution with its original bbox
      contributions.push({
        engine: secondaryEngine,
        text: match.text,
        confidence: match.confidence,
        bbox: match.bbox,  // Include secondary's original bbox
      });

      matchType = match.matchType;
      iouScore = match.iouScore;
      textAgreed = match.textMatches;
    }

    // Vote on final text
    const textResult = pickBestText(votes);
    const maxConf = Math.max(...votes.map((v) => v.confidence));

    // Check if ANY OCR engine has high confidence (>= 80%)
    const hasHighConfidence = votes.some((v) => v.confidence >= 80);

    // Build harmonize details only if multiple sources
    let harmonize: HarmonizeDetails | undefined;
    if (match) {
      harmonize = {
        matchType,
        iouScore,
        textAgreed,
        contributions,
        chosenTextReason: textResult.reason,
      };
    }

    const hw: ExtractedWord = {
      page: primary.page,
      wordId: -1, // Will be assigned later
      text: textResult.text,
      bbox: primary.bbox,
      engine: primaryEngine, // Use primary engine since bbox comes from it
      confidence: maxConf,
      sources: source.split('').map((c) =>
        c === 'T' ? 'tesseract' : 'doctr'
      ),
      harmonize,
      lowConfidence: !hasHighConfidence,
    };

    result.push(hw);
    usedBboxes.push(primary.bbox);
  }

  // Phase 2: Add ALL unmatched secondary words (no confidence filtering)
  // Mark as lowConfidence if below 80%
  const LOW_CONF_THRESHOLD = 80;

  for (let i = 0; i < secondaryWords.length; i++) {
    if (matchedSecondary.has(i)) continue;

    const word = secondaryWords[i];

    // Skip if overlaps existing bbox
    const overlaps = usedBboxes.some(
      (bbox) => bboxIoU(word.bbox, bbox) >= DEFAULT_IOU_THRESHOLD
    );
    if (overlaps) continue;

    const hw: ExtractedWord = {
      page: word.page,
      wordId: -1,
      text: word.text,
      bbox: word.bbox,
      engine: secondaryEngine,
      confidence: word.confidence,
      sources: [secondaryEngine],
      lowConfidence: word.confidence < LOW_CONF_THRESHOLD,
      // No harmonize field - this word was only detected by one engine
    };

    result.push(hw);
    usedBboxes.push(word.bbox);
  }

  // Sort by position and assign IDs
  result.sort((a, b) => {
    if (a.page !== b.page) return a.page - b.page;
    if (Math.abs(a.bbox.y0 - b.bbox.y0) > 10) return a.bbox.y0 - b.bbox.y0;
    return a.bbox.x0 - b.bbox.x0;
  });

  for (let i = 0; i < result.length; i++) {
    result[i].wordId = i;
  }

  return result;
}

/**
 * Check if two words are likely duplicates based on position and text.
 */
export function isDuplicate(
  a: Word,
  b: Word,
  iouThreshold: number = 0.5
): boolean {
  if (a.page !== b.page) return false;

  // Check bbox overlap
  const iou = bboxIoU(a.bbox, b.bbox);
  if (iou >= iouThreshold) return true;

  // Check text match + proximity
  if (a.text.toLowerCase() === b.text.toLowerCase()) {
    const centerDist = Math.sqrt(
      Math.pow(bboxCenterX(a.bbox) - bboxCenterX(b.bbox), 2) +
        Math.pow(bboxCenterY(a.bbox) - bboxCenterY(b.bbox), 2)
    );
    if (centerDist < 20) return true;
  }

  return false;
}

/**
 * Simple deduplication for single-engine results.
 */
export function deduplicateWords(words: Word[]): Word[] {
  const result: Word[] = [];

  for (const word of words) {
    const isDup = result.some((existing) => isDuplicate(word, existing));
    if (!isDup) {
      result.push(word);
    }
  }

  return result;
}
