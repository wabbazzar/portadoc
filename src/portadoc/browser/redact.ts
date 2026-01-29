/**
 * Entity detection for browser-side redaction.
 * Mirrors Python implementation in src/portadoc/redact.py
 */

import { getBasePath } from './basePath';

export enum EntityType {
  NONE = '',
  NAME = 'NAME',
  DATE = 'DATE',
  CODE = 'CODE',
}

// DATE patterns
const DATE_PATTERNS = [
  // US date: 7/24/25, 12-31-2024, with optional trailing punctuation
  /^\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}[,.]?$/,
  // ISO date: 2024-12-31
  /^\d{4}-\d{2}-\d{2}$/,
];

// CODE patterns (numeric identifiers)
const CODE_PATTERNS = [
  // Phone: (555) 123-4567 or 555-123-4567 or 555.123.4567
  /^\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}$/,
  // SSN: 123-45-6789
  /^\d{3}-\d{2}-\d{4}$/,
  // Generic 5+ digit codes
  /^\d{5,}$/,
  // Alphanumeric codes: ABC12345, A1234567
  /^[A-Z]{1,3}\d{4,}$/,
  // MRN-style: 7829341 (7 digits)
  /^\d{7}$/,
];

// Exclusion patterns (false positives to avoid)
const EXCLUSION_PATTERNS = [
  // Time: 10:28, 2:30
  /^\d{1,2}:\d{2}$/,
  // Currency: $100, $1,000.00
  /^\$[\d,]+\.?\d*$/,
  // Percent: 50%, 3.5%
  /^\d+\.?\d*%$/,
  // Common abbreviations that look like codes
  /^(AM|PM|ID|MRN|DOB|SSN|EIN)$/i,
];

function matchesDate(text: string): boolean {
  return DATE_PATTERNS.some(p => p.test(text));
}

function matchesCode(text: string): boolean {
  return CODE_PATTERNS.some(p => p.test(text));
}

function matchesExclusion(text: string): boolean {
  return EXCLUSION_PATTERNS.some(p => p.test(text));
}

function stripPunctuation(text: string): string {
  return text.replace(/^[^\w]+|[^\w]+$/g, '');
}

export interface EntityDetectorOptions {
  detectNames?: boolean;
  detectDates?: boolean;
  detectCodes?: boolean;
}

export interface DetectionResult {
  entityType: EntityType;
  redact: boolean;
}

/**
 * Fast entity detector using regex and dictionary lookup.
 *
 * Detection order (most specific first):
 * 1. Check exclusions (time, currency, percent) -> skip
 * 2. Check date patterns -> DATE
 * 3. Check code patterns -> CODE
 * 4. Check names dictionary -> NAME
 */
export class EntityDetector {
  private names: Set<string> = new Set();
  private detectNames: boolean;
  private detectDates: boolean;
  private detectCodes: boolean;
  private loaded = false;

  constructor(options: EntityDetectorOptions = {}) {
    this.detectNames = options.detectNames ?? true;
    this.detectDates = options.detectDates ?? true;
    this.detectCodes = options.detectCodes ?? true;
  }

  /**
   * Load the names dictionary from the server.
   * Must be called before detect() if detectNames is true.
   */
  async loadNames(): Promise<void> {
    if (!this.detectNames) {
      this.loaded = true;
      return;
    }

    try {
      const basePath = getBasePath();
      const response = await fetch(`${basePath}data/dictionaries/us_names.txt`);
      if (!response.ok) {
        console.warn('Failed to load names dictionary, name detection disabled');
        this.detectNames = false;
        this.loaded = true;
        return;
      }

      const text = await response.text();
      const lines = text.split('\n');
      for (const line of lines) {
        const name = line.trim();
        if (name && !name.startsWith('#')) {
          this.names.add(name.toLowerCase());
        }
      }

      console.log(`Loaded ${this.names.size} names for entity detection`);
      this.loaded = true;
    } catch (err) {
      console.warn('Error loading names dictionary:', err);
      this.detectNames = false;
      this.loaded = true;
    }
  }

  /**
   * Detect entity type for a single word.
   */
  detect(text: string): DetectionResult {
    if (!text || !text.trim()) {
      return { entityType: EntityType.NONE, redact: false };
    }

    const trimmed = text.trim();

    // Check exclusions first (false positives)
    if (matchesExclusion(trimmed)) {
      return { entityType: EntityType.NONE, redact: false };
    }

    // Check dates
    if (this.detectDates && matchesDate(trimmed)) {
      return { entityType: EntityType.DATE, redact: true };
    }

    // Check codes
    if (this.detectCodes && matchesCode(trimmed)) {
      return { entityType: EntityType.CODE, redact: true };
    }

    // Check names (case-insensitive, strip punctuation)
    if (this.detectNames) {
      const cleanText = stripPunctuation(trimmed);
      if (cleanText && this.names.has(cleanText.toLowerCase())) {
        return { entityType: EntityType.NAME, redact: true };
      }
    }

    return { entityType: EntityType.NONE, redact: false };
  }

  /**
   * Detect entities for a batch of texts.
   */
  detectBatch(texts: string[]): DetectionResult[] {
    return texts.map(text => this.detect(text));
  }

  /**
   * Check if the detector is ready (names loaded).
   */
  isLoaded(): boolean {
    return this.loaded;
  }

  /**
   * Get the count of loaded names.
   */
  getNameCount(): number {
    return this.names.size;
  }
}
