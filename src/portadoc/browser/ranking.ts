/**
 * Multi-signal ranking for OCR correction candidates.
 * TypeScript port of src/portadoc/ranking.py
 */

export interface RankingConfig {
  enabled: boolean;
  weight: number;
}

export interface FrequencyConfig extends RankingConfig {
  source: string;
  fallbackFrequency: number;
}

export interface DocumentConfig extends RankingConfig {
  minOccurrences: number;
}

export interface BigramConfig extends RankingConfig {
  source: string;
  window: number;
}

export interface OCRModelConfig extends RankingConfig {
  source: string;
}

interface Confusion {
  pattern: string;
  confused_with: string[];
  probability: number;
  context?: 'end' | 'middle';
}

export class FrequencyRanker {
  private config: FrequencyConfig;
  private frequencies: Map<string, number> = new Map();
  private maxFrequency = 1;

  constructor(config: FrequencyConfig) {
    this.config = config;
  }

  async load(): Promise<void> {
    if (!this.config.enabled) return;

    try {
      const response = await fetch(this.config.source);
      if (!response.ok) {
        console.warn(`Failed to load frequency data: ${response.statusText}`);
        return;
      }
      const data: Record<string, number> = await response.json();

      for (const [word, freq] of Object.entries(data)) {
        this.frequencies.set(word.toLowerCase(), freq);
        this.maxFrequency = Math.max(this.maxFrequency, freq);
      }

      console.log(`Loaded ${this.frequencies.size} word frequencies`);
    } catch (err) {
      console.error('Failed to load frequency data:', err);
    }
  }

  getFrequencyFactor(word: string): number {
    if (!this.config.enabled || this.frequencies.size === 0) {
      return 1.0;
    }

    const freq = this.frequencies.get(word.toLowerCase()) ?? this.config.fallbackFrequency;

    // Log normalization to compress range
    const factor = Math.log10(freq + 1) / Math.log10(this.maxFrequency + 1);

    // Apply weight
    return this.config.weight !== 1.0 ? Math.pow(factor, 1.0 / this.config.weight) : factor;
  }
}

export class DocumentRanker {
  private config: DocumentConfig;
  private documentIndex: Map<string, number> = new Map();

  constructor(config: DocumentConfig) {
    this.config = config;
  }

  buildDocumentIndex(words: string[]): void {
    this.documentIndex.clear();
    for (const word of words) {
      const key = word.toLowerCase();
      this.documentIndex.set(key, (this.documentIndex.get(key) ?? 0) + 1);
    }
  }

  getDocumentFactor(word: string): number {
    if (!this.config.enabled) {
      return 1.0;
    }

    const count = this.documentIndex.get(word.toLowerCase()) ?? 0;

    // Only boost if meets minimum threshold
    if (count < this.config.minOccurrences) {
      return 1.0;
    }

    // Logarithmic boost
    const boost = 1.0 + this.config.weight * Math.log10(count + 1);
    return boost;
  }
}

export class BigramRanker {
  private config: BigramConfig;
  private bigrams: Map<string, number> = new Map();
  private unigramCounts: Map<string, number> = new Map();

  constructor(config: BigramConfig) {
    this.config = config;
  }

  async load(): Promise<void> {
    if (!this.config.enabled) return;

    try {
      const response = await fetch(this.config.source);
      if (!response.ok) {
        console.warn(`Failed to load bigram data: ${response.statusText}`);
        return;
      }
      const data: Record<string, number> = await response.json();

      for (const [words, freq] of Object.entries(data)) {
        const parts = words.split(' ');
        if (parts.length === 2) {
          const [w1, w2] = parts;
          this.bigrams.set(`${w1.toLowerCase()} ${w2.toLowerCase()}`, freq);

          // Track unigram counts for probability calculation
          const w1Lower = w1.toLowerCase();
          const w2Lower = w2.toLowerCase();
          this.unigramCounts.set(w1Lower, (this.unigramCounts.get(w1Lower) ?? 0) + freq);
          this.unigramCounts.set(w2Lower, (this.unigramCounts.get(w2Lower) ?? 0) + freq);
        }
      }

      console.log(`Loaded ${this.bigrams.size} bigrams`);
    } catch (err) {
      console.error('Failed to load bigram data:', err);
    }
  }

  getBigramFactor(word: string, prevWord?: string, nextWord?: string): number {
    if (!this.config.enabled || this.bigrams.size === 0) {
      return 1.0;
    }

    const wordLower = word.toLowerCase();
    const scores: number[] = [];

    // P(word | prevWord)
    if (prevWord) {
      const prevLower = prevWord.toLowerCase();
      const bigramCount = this.bigrams.get(`${prevLower} ${wordLower}`) ?? 0;
      const unigramCount = this.unigramCounts.get(prevLower) ?? 0;
      if (unigramCount > 0) {
        const prob = bigramCount / unigramCount;
        scores.push(prob);
      }
    }

    // P(nextWord | word)
    if (nextWord) {
      const nextLower = nextWord.toLowerCase();
      const bigramCount = this.bigrams.get(`${wordLower} ${nextLower}`) ?? 0;
      const unigramCount = this.unigramCounts.get(wordLower) ?? 0;
      if (unigramCount > 0) {
        const prob = bigramCount / unigramCount;
        scores.push(prob);
      }
    }

    if (scores.length === 0) {
      return 1.0;
    }

    // Average probability, normalized and weighted
    const avgProb = scores.reduce((a, b) => a + b, 0) / scores.length;

    // Convert to factor (probabilities are very small, so use log scale)
    if (avgProb > 0) {
      const factor = 1.0 + this.config.weight * Math.log10(avgProb * 1000000 + 1) / 10;
      return Math.max(0.5, Math.min(2.0, factor)); // Clamp to reasonable range
    }

    return 1.0;
  }
}

export class OCRErrorModel {
  private config: OCRModelConfig;
  private confusions: Confusion[] = [];

  constructor(config: OCRModelConfig) {
    this.config = config;
  }

  async load(): Promise<void> {
    if (!this.config.enabled) return;

    try {
      const response = await fetch(this.config.source);
      if (!response.ok) {
        console.warn(`Failed to load OCR confusion data: ${response.statusText}`);
        return;
      }
      const data = await response.json();
      this.confusions = data.confusions ?? [];

      console.log(`Loaded ${this.confusions.length} OCR confusion patterns`);
    } catch (err) {
      console.error('Failed to load OCR confusion data:', err);
    }
  }

  getOcrFactor(original: string, candidate: string): number {
    if (!this.config.enabled || this.confusions.length === 0) {
      return 1.0;
    }

    const factors: number[] = [];

    // Check for direct substitutions
    const minLen = Math.min(original.length, candidate.length);
    for (let i = 0; i < minLen; i++) {
      const oChar = original[i];
      const cChar = candidate[i];
      if (oChar !== cChar) {
        const factor = this.checkConfusion(oChar, cChar, i, original.length);
        if (factor > 1.0) {
          factors.push(factor);
        }
      }
    }

    // Check for insertions/deletions (pattern-based)
    if (original.length !== candidate.length) {
      for (const confusion of this.confusions) {
        const pattern = confusion.pattern;
        if (pattern.length > 1) {
          for (const confused of confusion.confused_with) {
            if ((original.includes(pattern) && candidate.includes(confused)) ||
                (original.includes(confused) && candidate.includes(pattern))) {
              factors.push(confusion.probability);
            }
          }
        }
      }
    }

    if (factors.length === 0) {
      return 1.0;
    }

    // Combine factors (geometric mean to avoid over-boosting)
    const product = factors.reduce((a, b) => a * b, 1.0);
    let result = Math.pow(product, 1.0 / factors.length);

    // Apply weight
    if (this.config.weight !== 1.0) {
      result = 1.0 + (result - 1.0) * this.config.weight;
    }

    return result;
  }

  private checkConfusion(char1: string, char2: string, position: number, wordLen: number): number {
    for (const confusion of this.confusions) {
      const pattern = confusion.pattern;
      const confusedWith = confusion.confused_with;
      const probability = confusion.probability;

      // Check if single-character pattern
      if (pattern.length === 1) {
        // Check context if specified
        const context = confusion.context;
        if (context === 'end' && position !== wordLen - 1) {
          continue;
        }
        if (context === 'middle' && (position === 0 || position === wordLen - 1)) {
          continue;
        }

        // Check both directions
        if ((pattern === char1 && confusedWith.includes(char2)) ||
            (pattern === char2 && confusedWith.includes(char1))) {
          return 1.0 + probability;
        }
      }
    }

    return 1.0;
  }
}

export class MultiSignalRanker {
  private frequencyRanker: FrequencyRanker;
  private documentRanker: DocumentRanker;
  private bigramRanker: BigramRanker;
  private ocrRanker: OCRErrorModel;

  constructor(
    frequencyConfig?: FrequencyConfig,
    documentConfig?: DocumentConfig,
    bigramConfig?: BigramConfig,
    ocrConfig?: OCRModelConfig
  ) {
    this.frequencyRanker = new FrequencyRanker(frequencyConfig ?? {
      enabled: true,
      weight: 1.0,
      source: '/public/data/frequencies.json',
      fallbackFrequency: 1,
    });
    this.documentRanker = new DocumentRanker(documentConfig ?? {
      enabled: true,
      weight: 0.3,
      minOccurrences: 2,
    });
    this.bigramRanker = new BigramRanker(bigramConfig ?? {
      enabled: true,
      weight: 0.5,
      source: '/public/data/bigrams.json',
      window: 1,
    });
    this.ocrRanker = new OCRErrorModel(ocrConfig ?? {
      enabled: true,
      weight: 0.4,
      source: '/public/data/ocr_confusions.json',
    });
  }

  async load(): Promise<void> {
    await Promise.all([
      this.frequencyRanker.load(),
      this.bigramRanker.load(),
      this.ocrRanker.load(),
    ]);
  }

  buildDocumentIndex(words: string[]): void {
    this.documentRanker.buildDocumentIndex(words);
  }

  rankCandidate(
    original: string,
    candidate: string,
    prevWord?: string,
    nextWord?: string
  ): [number, number, number, number] {
    const freqFactor = this.frequencyRanker.getFrequencyFactor(candidate);
    const docFactor = this.documentRanker.getDocumentFactor(candidate);
    const bigramFactor = this.bigramRanker.getBigramFactor(candidate, prevWord, nextWord);
    const ocrFactor = this.ocrRanker.getOcrFactor(original, candidate);

    return [freqFactor, docFactor, bigramFactor, ocrFactor];
  }

  getCompositeFactor(
    original: string,
    candidate: string,
    prevWord?: string,
    nextWord?: string
  ): number {
    const [freq, doc, bigram, ocr] = this.rankCandidate(original, candidate, prevWord, nextWord);
    return freq * doc * bigram * ocr;
  }
}
