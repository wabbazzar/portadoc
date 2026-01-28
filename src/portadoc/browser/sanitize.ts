/**
 * OCR Text Sanitization using dictionary-based Levenshtein distance matching.
 * Port of src/portadoc/sanitize.py
 */

import { MultiSignalRanker } from './ranking.js';
import { resolveAssetPath } from './basePath.js';

export enum SanitizeStatus {
  SKIPPED = 'skipped',
  PRESERVED = 'preserved',
  CORRECTED = 'corrected',
  CONTEXT_CORRECTED = 'context',
  UNCERTAIN = 'uncertain',
}

export interface SanitizeResult {
  originalText: string;
  sanitizedText: string;
  status: SanitizeStatus;
  confidence: number;
  editDistance: number;
  correctionScore: number;
  matchedDictionary?: string;
  frequencyFactor?: number;
  documentFactor?: number;
  bigramFactor?: number;
  ocrFactor?: number;
}

export interface RankingSignalConfig {
  enabled: boolean;
  weight: number;
}

export interface SanitizeConfig {
  skip: {
    pixelDetector: boolean;
    singleCharMaxConf: number;
  };
  preserve: {
    confidenceThreshold: number;
    numericRatio: number;
    exactMatchDictionaries: string[];
  };
  correct: {
    maxEditDistance: number;
    minCorrectionScore: number;
    dictionaryWeights: Record<string, number>;
  };
  context: {
    enabled: boolean;
    maxEditDistance: number;
    minContextScore: number;
    ngramWindow: number;
  };
  ranking?: {
    frequency?: RankingSignalConfig & { source?: string; fallbackFrequency?: number };
    document?: RankingSignalConfig & { minOccurrences?: number };
    bigram?: RankingSignalConfig & { source?: string; window?: number };
    ocrModel?: RankingSignalConfig & { source?: string };
  };
}

const DEFAULT_CONFIG: SanitizeConfig = {
  skip: {
    pixelDetector: true,
    singleCharMaxConf: 50,
  },
  preserve: {
    confidenceThreshold: 100,
    numericRatio: 0.5,
    exactMatchDictionaries: ['english', 'names', 'medical', 'custom'],
  },
  correct: {
    maxEditDistance: 2,
    minCorrectionScore: 0.3,
    dictionaryWeights: {
      english: 1.0,
      names: 0.9,
      medical: 1.2,
      custom: 1.5,
    },
  },
  context: {
    enabled: true,
    maxEditDistance: 3,
    minContextScore: 0.2,
    ngramWindow: 2,
  },
};

interface FuzzyMatch {
  word: string;
  distance: number;
  dictionary: string;
  score: number;
  frequencyFactor?: number;
  documentFactor?: number;
  bigramFactor?: number;
  ocrFactor?: number;
}

/**
 * Levenshtein edit distance between two strings.
 * Reused from validate.ts
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

export class Sanitizer {
  private config: SanitizeConfig;
  private dictionaries: Map<string, Set<string>> = new Map();
  private allWords: Set<string> = new Set();
  // Multi-level index: dictionary -> first char -> length -> words
  // This reduces search space from O(n) to O(n/26/L) on average
  private indexedWords: Map<string, Map<string, Map<number, string[]>>> = new Map();
  // Result cache for repeated words
  private resultCache: Map<string, SanitizeResult> = new Map();
  // Fuzzy match cache (text|maxDistance -> results)
  private fuzzyCache: Map<string, FuzzyMatch[]> = new Map();
  private loaded = false;
  private ranker: MultiSignalRanker | null = null;

  private numericPattern = /\d/g;
  private emailPattern = /@/;
  private datePattern = /^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}[,.]?$/;

  constructor(config?: Partial<SanitizeConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    // Initialize ranker if ranking config provided
    if (this.config.ranking) {
      const r = this.config.ranking;
      this.ranker = new MultiSignalRanker(
        r.frequency ? {
          enabled: r.frequency.enabled ?? true,
          weight: r.frequency.weight ?? 1.0,
          source: r.frequency.source ?? '/public/data/frequencies.json',
          fallbackFrequency: r.frequency.fallbackFrequency ?? 1,
        } : undefined,
        r.document ? {
          enabled: r.document.enabled ?? true,
          weight: r.document.weight ?? 0.3,
          minOccurrences: r.document.minOccurrences ?? 2,
        } : undefined,
        r.bigram ? {
          enabled: r.bigram.enabled ?? true,
          weight: r.bigram.weight ?? 0.5,
          source: r.bigram.source ?? '/public/data/bigrams.json',
          window: r.bigram.window ?? 1,
        } : undefined,
        r.ocrModel ? {
          enabled: r.ocrModel.enabled ?? true,
          weight: r.ocrModel.weight ?? 0.4,
          source: r.ocrModel.source ?? '/public/data/ocr_confusions.json',
        } : undefined
      );
    }
  }

  async loadDictionaries(): Promise<void> {
    if (this.loaded) return;

    // Load ranker data if enabled
    if (this.ranker) {
      await this.ranker.load();
    }

    const names = ['english', 'names', 'medical', 'custom'];
    const loads = names.map(async (name) => {
      try {
        const response = await fetch(resolveAssetPath(`dictionaries/${name}.json`));
        if (!response.ok) {
          console.warn(`Failed to load ${name}.json: ${response.statusText}`);
          return;
        }
        const words: string[] = await response.json();
        const wordSet = new Set(words);

        // Add both original and lowercase for case-insensitive matching
        words.forEach(w => {
          this.allWords.add(w);
          this.allWords.add(w.toLowerCase());
        });

        // Build multi-level index: first char -> length -> words
        const charIndex = new Map<string, Map<number, string[]>>();
        for (const word of words) {
          const lower = word.toLowerCase();
          const firstChar = lower[0] || '_';
          const len = word.length;

          if (!charIndex.has(firstChar)) {
            charIndex.set(firstChar, new Map());
          }
          const lenMap = charIndex.get(firstChar)!;
          if (!lenMap.has(len)) {
            lenMap.set(len, []);
          }
          lenMap.get(len)!.push(word);
        }
        this.indexedWords.set(name, charIndex);

        this.dictionaries.set(name, wordSet);
        console.log(`Loaded ${name}: ${words.length} words`);
      } catch (err) {
        console.error(`Error loading ${name}.json:`, err);
      }
    });

    await Promise.all(loads);
    this.loaded = true;
  }

  sanitizeWord(text: string, confidence: number, engine?: string): SanitizeResult {
    if (!this.loaded) {
      throw new Error('Dictionaries not loaded. Call loadDictionaries() first.');
    }

    // Check cache first (significant speedup for repeated words)
    const cacheKey = `${text}|${confidence}|${engine || ''}`;
    const cached = this.resultCache.get(cacheKey);
    if (cached) return cached;

    // Phase 0: Skip
    const skipResult = this.phaseSkip(text, confidence, engine);
    if (skipResult) {
      this.resultCache.set(cacheKey, skipResult);
      return skipResult;
    }

    // Phase 1: Preserve
    const preserveResult = this.phasePreserve(text, confidence);
    if (preserveResult) {
      this.resultCache.set(cacheKey, preserveResult);
      return preserveResult;
    }

    // Phase 2: Correct
    const correctResult = this.phaseCorrect(text, confidence);
    if (correctResult) {
      this.resultCache.set(cacheKey, correctResult);
      return correctResult;
    }

    // Phase 4: Uncertain
    const uncertainResult: SanitizeResult = {
      originalText: text,
      sanitizedText: text,
      status: SanitizeStatus.UNCERTAIN,
      confidence,
      editDistance: 0,
      correctionScore: 0,
    };
    this.resultCache.set(cacheKey, uncertainResult);
    return uncertainResult;
  }

  sanitizeWords(words: Array<{ text: string; confidence: number; engine?: string }>): SanitizeResult[] {
    // Build document index for ranking
    if (this.ranker) {
      const wordTexts = words.map(w => w.text);
      this.ranker.buildDocumentIndex(wordTexts);
    }

    const results: SanitizeResult[] = [];

    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      const prevWord = i > 0 ? words[i - 1].text : undefined;
      const nextWord = i < words.length - 1 ? words[i + 1].text : undefined;

      let result = this.sanitizeWordWithContext(word.text, word.confidence, word.engine, prevWord, nextWord);

      // Phase 3: Context correction for uncertain words
      if (result.status === SanitizeStatus.UNCERTAIN && this.config.context.enabled) {
        const contextResult = this.phaseContext(word.text, word.confidence, words, i);
        if (contextResult) {
          result = contextResult;
        }
      }

      results.push(result);
    }

    return results;
  }

  private sanitizeWordWithContext(
    text: string,
    confidence: number,
    engine?: string,
    prevWord?: string,
    nextWord?: string
  ): SanitizeResult {
    if (!this.loaded) {
      throw new Error('Dictionaries not loaded. Call loadDictionaries() first.');
    }

    // Phase 0: Skip
    const skipResult = this.phaseSkip(text, confidence, engine);
    if (skipResult) return skipResult;

    // Phase 1: Preserve
    const preserveResult = this.phasePreserve(text, confidence);
    if (preserveResult) return preserveResult;

    // Phase 2: Correct with ranking
    const correctResult = this.phaseCorrectWithRanking(text, confidence, prevWord, nextWord);
    if (correctResult) return correctResult;

    // Phase 4: Uncertain
    return {
      originalText: text,
      sanitizedText: text,
      status: SanitizeStatus.UNCERTAIN,
      confidence,
      editDistance: 0,
      correctionScore: 0,
    };
  }

  private phaseSkip(text: string, confidence: number, engine?: string): SanitizeResult | null {
    // Skip pixel detector results
    if (this.config.skip.pixelDetector && engine === 'pixel_detector') {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.SKIPPED,
        confidence,
        editDistance: 0,
        correctionScore: 0,
      };
    }

    // Skip low-confidence single characters
    if (text.length === 1 && confidence < this.config.skip.singleCharMaxConf) {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.SKIPPED,
        confidence,
        editDistance: 0,
        correctionScore: 0,
      };
    }

    return null;
  }

  private phasePreserve(text: string, confidence: number): SanitizeResult | null {
    // Preserve high-confidence words
    if (confidence >= this.config.preserve.confidenceThreshold) {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.PRESERVED,
        confidence,
        editDistance: 0,
        correctionScore: 1.0,
      };
    }

    // Preserve words with high numeric ratio
    const numericRatio = this.getNumericRatio(text);
    if (numericRatio >= this.config.preserve.numericRatio) {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.PRESERVED,
        confidence,
        editDistance: 0,
        correctionScore: 1.0,
      };
    }

    // Preserve emails
    if (this.emailPattern.test(text)) {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.PRESERVED,
        confidence,
        editDistance: 0,
        correctionScore: 1.0,
      };
    }

    // Preserve dates
    if (this.datePattern.test(text)) {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.PRESERVED,
        confidence,
        editDistance: 0,
        correctionScore: 1.0,
      };
    }

    // Preserve exact dictionary matches
    const matchedDict = this.exactMatch(text);
    if (matchedDict) {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.PRESERVED,
        confidence,
        editDistance: 0,
        correctionScore: 1.0,
        matchedDictionary: matchedDict,
      };
    }

    // Try stripping punctuation
    const stripped = text.replace(/^[.,;:!?()\[\]{}\"']+|[.,;:!?()\[\]{}\"']+$/g, '');
    if (stripped !== text && stripped) {
      const matchedDictStripped = this.exactMatch(stripped);
      if (matchedDictStripped) {
        return {
          originalText: text,
          sanitizedText: text,
          status: SanitizeStatus.PRESERVED,
          confidence,
          editDistance: 0,
          correctionScore: 1.0,
          matchedDictionary: matchedDictStripped,
        };
      }
    }

    return null;
  }

  private phaseCorrect(text: string, confidence: number): SanitizeResult | null {
    return this.phaseCorrectWithRanking(text, confidence, undefined, undefined);
  }

  private phaseCorrectWithRanking(
    text: string,
    confidence: number,
    prevWord?: string,
    nextWord?: string
  ): SanitizeResult | null {
    // Strip punctuation
    const stripped = text.replace(/^[.,;:!?()\[\]{}\"']+|[.,;:!?()\[\]{}\"']+$/g, '');
    const leadingMatch = text.match(/^[.,;:!?()\[\]{}\"']+/);
    const trailingMatch = text.match(/[.,;:!?()\[\]{}\"']+$/);
    const prefix = leadingMatch ? leadingMatch[0] : '';
    const suffix = trailingMatch ? trailingMatch[0] : '';

    if (!stripped) return null;

    // Find fuzzy matches with ranking
    const matches = this.fuzzyMatchWithRanking(
      stripped,
      this.config.correct.maxEditDistance,
      prevWord,
      nextWord
    );
    if (matches.length === 0) return null;

    const best = matches[0];

    // Check score threshold
    if (best.score < this.config.correct.minCorrectionScore) return null;

    // Don't correct if distance is 0
    if (best.distance === 0) {
      return {
        originalText: text,
        sanitizedText: text,
        status: SanitizeStatus.PRESERVED,
        confidence,
        editDistance: 0,
        correctionScore: best.score,
        matchedDictionary: best.dictionary,
        frequencyFactor: best.frequencyFactor,
        documentFactor: best.documentFactor,
        bigramFactor: best.bigramFactor,
        ocrFactor: best.ocrFactor,
      };
    }

    // Preserve case pattern
    const corrected = this.preserveCase(stripped, best.word);
    const sanitized = prefix + corrected + suffix;

    return {
      originalText: text,
      sanitizedText: sanitized,
      status: SanitizeStatus.CORRECTED,
      confidence,
      editDistance: best.distance,
      correctionScore: best.score,
      matchedDictionary: best.dictionary,
      frequencyFactor: best.frequencyFactor,
      documentFactor: best.documentFactor,
      bigramFactor: best.bigramFactor,
      ocrFactor: best.ocrFactor,
    };
  }

  private phaseContext(
    text: string,
    confidence: number,
    words: Array<{ text: string; confidence: number; engine?: string }>,
    index: number
  ): SanitizeResult | null {
    // Strip punctuation
    const stripped = text.replace(/^[.,;:!?()\[\]{}\"']+|[.,;:!?()\[\]{}\"']+$/g, '');
    const leadingMatch = text.match(/^[.,;:!?()\[\]{}\"']+/);
    const trailingMatch = text.match(/[.,;:!?()\[\]{}\"']+$/);
    const prefix = leadingMatch ? leadingMatch[0] : '';
    const suffix = trailingMatch ? trailingMatch[0] : '';

    if (!stripped) return null;

    // Get prev/next words for ranking
    const prevWord = index > 0 ? words[index - 1].text : undefined;
    const nextWord = index < words.length - 1 ? words[index + 1].text : undefined;

    // Extract context window (adjacent words)
    const window = this.config.context.ngramWindow;
    const start = Math.max(0, index - window);
    const end = Math.min(words.length, index + window + 1);

    // Build context (exclude current word)
    const contextWords: string[] = [];
    for (let i = start; i < end; i++) {
      if (i !== index) {
        contextWords.push(words[i].text.toLowerCase());
      }
    }

    // Try with higher edit distance and ranking
    const matches = this.fuzzyMatchWithRanking(
      stripped,
      this.config.context.maxEditDistance,
      prevWord,
      nextWord
    );
    if (matches.length === 0) return null;

    // Boost scores if candidate appears in context words
    // (For now, simple presence check. Future: use n-gram frequencies)
    const boostedMatches = matches.map((match) => {
      const matchLower = match.word.toLowerCase();
      const inContext = contextWords.some((ctx) => ctx.includes(matchLower) || matchLower.includes(ctx));
      return {
        ...match,
        score: inContext ? match.score * 1.5 : match.score,
      };
    });

    // Re-sort after boosting
    boostedMatches.sort((a, b) => b.score - a.score);
    const best = boostedMatches[0];

    if (best.score < this.config.context.minContextScore) return null;
    if (best.distance === 0) return null;

    // Preserve case and punctuation
    const corrected = this.preserveCase(stripped, best.word);
    const sanitized = prefix + corrected + suffix;

    return {
      originalText: text,
      sanitizedText: sanitized,
      status: SanitizeStatus.CONTEXT_CORRECTED,
      confidence,
      editDistance: best.distance,
      correctionScore: best.score,
      matchedDictionary: best.dictionary,
      frequencyFactor: best.frequencyFactor,
      documentFactor: best.documentFactor,
      bigramFactor: best.bigramFactor,
      ocrFactor: best.ocrFactor,
    };
  }

  private exactMatch(word: string): string | null {
    // Check case-sensitive first
    if (this.allWords.has(word)) {
      for (const name of this.config.preserve.exactMatchDictionaries) {
        const dict = this.dictionaries.get(name);
        if (dict && dict.has(word)) {
          return name;
        }
      }
    }

    // Check case-insensitive
    const lower = word.toLowerCase();
    if (this.allWords.has(lower)) {
      for (const name of this.config.preserve.exactMatchDictionaries) {
        const dict = this.dictionaries.get(name);
        if (dict) {
          for (const dictWord of dict) {
            if (dictWord.toLowerCase() === lower) {
              return name;
            }
          }
        }
      }
    }

    return null;
  }

  private fuzzyMatchWithRanking(
    word: string,
    maxDistance: number,
    prevWord?: string,
    nextWord?: string
  ): FuzzyMatch[] {
    const lookup = word.toLowerCase();

    // Check fuzzy cache first (only if no ranking)
    const cacheKey = `${lookup}|${maxDistance}`;
    if (!this.ranker) {
      const cached = this.fuzzyCache.get(cacheKey);
      if (cached) return cached;
    }

    const wordLen = lookup.length;
    const firstChar = lookup[0];

    // Only compare words within valid length range
    const minLen = Math.max(1, wordLen - maxDistance);
    const maxLen = wordLen + maxDistance;

    // Fast path: only check words with same first char
    // This covers most cases since first char is usually correct
    let results = this.fuzzyMatchWithChars(lookup, [firstChar], minLen, maxLen, maxDistance);

    // If we found a good match (distance 0 or 1), return immediately
    if (results.length > 0 && results[0].distance <= 1) {
      this.fuzzyCache.set(cacheKey, results);
      return results;
    }

    // Slow path: check second char of word (handles first char deletion)
    // and all adjacent chars (handles first char substitution)
    if (maxDistance >= 1 && results.length === 0) {
      const extraChars: string[] = [];
      if (lookup.length > 1) {
        extraChars.push(lookup[1]); // Second char (first char deleted)
      }
      // Adjacent chars in alphabet
      const code = firstChar.charCodeAt(0);
      if (code > 97) extraChars.push(String.fromCharCode(code - 1));
      if (code < 122) extraChars.push(String.fromCharCode(code + 1));

      const extraResults = this.fuzzyMatchWithChars(lookup, extraChars, minLen, maxLen, maxDistance);
      results = results.concat(extraResults);
    }

    // Apply ranking if enabled
    if (this.ranker) {
      for (const match of results) {
        const [freqFactor, docFactor, bigramFactor, ocrFactor] = this.ranker.rankCandidate(
          word,
          match.word,
          prevWord,
          nextWord
        );
        match.frequencyFactor = freqFactor;
        match.documentFactor = docFactor;
        match.bigramFactor = bigramFactor;
        match.ocrFactor = ocrFactor;
        // Apply factors to score
        match.score = match.score * freqFactor * docFactor * bigramFactor * ocrFactor;
      }
    }

    // Sort by score descending
    results.sort((a, b) => b.score - a.score);

    // Only cache if no ranking (ranking is context-dependent)
    if (!this.ranker) {
      this.fuzzyCache.set(cacheKey, results);
    }

    return results;
  }

  private fuzzyMatchWithChars(
    lookup: string,
    chars: string[],
    minLen: number,
    maxLen: number,
    maxDistance: number
  ): FuzzyMatch[] {
    const results: FuzzyMatch[] = [];
    let foundExact = false;

    for (const [name, charIndex] of this.indexedWords) {
      if (foundExact) break; // Early termination for exact matches
      const weight = this.config.correct.dictionaryWeights[name] ?? 1.0;

      for (const char of chars) {
        if (foundExact) break;
        const lenMap = charIndex.get(char);
        if (!lenMap) continue;

        for (let len = minLen; len <= maxLen; len++) {
          if (foundExact) break;
          const wordsAtLen = lenMap.get(len);
          if (!wordsAtLen) continue;

          for (const dictWord of wordsAtLen) {
            const compare = dictWord.toLowerCase();
            const dist = levenshtein(lookup, compare);

            if (dist <= maxDistance) {
              const score = weight * (1 - dist / (maxDistance + 1));
              results.push({ word: dictWord, distance: dist, dictionary: name, score });

              // Early termination: if we found exact match, stop searching
              if (dist === 0) {
                foundExact = true;
                break;
              }
            }
          }
        }
      }
    }

    return results;
  }

  private getNumericRatio(text: string): number {
    if (!text) return 0;
    const matches = text.match(this.numericPattern);
    return matches ? matches.length / text.length : 0;
  }

  private preserveCase(original: string, corrected: string): string {
    if (!original || !corrected) return corrected;

    // All uppercase
    if (original === original.toUpperCase()) {
      return corrected.toUpperCase();
    }

    // All lowercase
    if (original === original.toLowerCase()) {
      return corrected.toLowerCase();
    }

    // Title case
    if (original[0] === original[0].toUpperCase() && original.slice(1) === original.slice(1).toLowerCase()) {
      return corrected[0].toUpperCase() + corrected.slice(1).toLowerCase();
    }

    // Mixed case - just return corrected
    return corrected;
  }
}
