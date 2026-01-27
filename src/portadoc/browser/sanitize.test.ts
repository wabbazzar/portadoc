import { describe, it, expect, beforeAll } from 'vitest';
import { Sanitizer, SanitizeStatus } from './sanitize';
import * as fs from 'fs';
import * as path from 'path';

describe('Sanitizer', () => {
  let sanitizer: Sanitizer;

  beforeAll(async () => {
    // Mock fetch for Node.js environment
    const dictPath = path.join(__dirname, 'public', 'dictionaries');
    global.fetch = async (url: string | URL | Request): Promise<Response> => {
      const urlStr = url.toString();
      const filename = urlStr.split('/').pop();
      const filePath = path.join(dictPath, filename!);

      if (fs.existsSync(filePath)) {
        const data = fs.readFileSync(filePath, 'utf-8');
        return {
          ok: true,
          json: async () => JSON.parse(data),
          statusText: 'OK',
        } as Response;
      } else {
        return {
          ok: false,
          statusText: 'Not Found',
        } as Response;
      }
    };

    sanitizer = new Sanitizer();
    await sanitizer.loadDictionaries();
  });

  describe('Phase 2: Single-word corrections', () => {
    it('corrects Decument -> Document (distance 1)', () => {
      const result = sanitizer.sanitizeWord('Decument', 60);
      expect(result.sanitizedText.toLowerCase()).toBe('document');
      expect(result.status).toBe(SanitizeStatus.CORRECTED);
      expect(result.editDistance).toBe(1);
    });

    it('corrects Compassianae -> Compassionate (distance 2)', () => {
      const result = sanitizer.sanitizeWord('Compassianae', 60);
      expect(result.sanitizedText.toLowerCase()).toBe('compassionate');
      expect(result.status).toBe(SanitizeStatus.CORRECTED);
      expect(result.editDistance).toBeLessThanOrEqual(2);
    });

    it('corrects Speties: -> Species: (distance 2)', () => {
      const result = sanitizer.sanitizeWord('Speties:', 60);
      expect(result.sanitizedText.toLowerCase()).toBe('species:');
      expect([SanitizeStatus.CORRECTED, SanitizeStatus.PRESERVED]).toContain(result.status);
    });

    it('corrects Domeelic -> Domestic (distance 2)', () => {
      const result = sanitizer.sanitizeWord('Domeelic', 60);
      expect(result.sanitizedText.toLowerCase()).toBe('domestic');
      expect(result.status).toBe(SanitizeStatus.CORRECTED);
      expect(result.editDistance).toBeLessThanOrEqual(2);
    });
  });

  describe('Phase 1: Preservation', () => {
    it('preserves valid word Cars', () => {
      const result = sanitizer.sanitizeWord('Cars', 80);
      expect(result.sanitizedText).toBe('Cars');
      expect(result.status).toBe(SanitizeStatus.PRESERVED);
    });

    it('preserves numeric values', () => {
      const result = sanitizer.sanitizeWord('555-0123', 70);
      expect(result.sanitizedText).toBe('555-0123');
      expect(result.status).toBe(SanitizeStatus.PRESERVED);
    });

    it('preserves email addresses', () => {
      const result = sanitizer.sanitizeWord('test@example.com', 70);
      expect(result.sanitizedText).toBe('test@example.com');
      expect(result.status).toBe(SanitizeStatus.PRESERVED);
    });

    it('preserves dates', () => {
      const result = sanitizer.sanitizeWord('01/15/2025', 70);
      expect(result.sanitizedText).toBe('01/15/2025');
      expect(result.status).toBe(SanitizeStatus.PRESERVED);
    });

    it('preserves high-confidence words even if not in dictionary', () => {
      const result = sanitizer.sanitizeWord('XYZ123', 100);
      expect(result.sanitizedText).toBe('XYZ123');
      expect(result.status).toBe(SanitizeStatus.PRESERVED);
    });
  });

  describe('Phase 3: Context corrections', () => {
    it('corrects Folinn -> Feline (may correct in phase 2 or 3)', () => {
      const words = [
        { text: 'for', confidence: 80, engine: 'tesseract' },
        { text: 'our', confidence: 80, engine: 'tesseract' },
        { text: 'Folinn', confidence: 60, engine: 'tesseract' },
        { text: 'Friends', confidence: 80, engine: 'tesseract' },
      ];
      const results = sanitizer.sanitizeWords(words);
      expect(results[2].sanitizedText.toLowerCase()).toBe('feline');
      // May be corrected in phase 2 or context phase 3, both acceptable
      expect([SanitizeStatus.CORRECTED, SanitizeStatus.CONTEXT_CORRECTED]).toContain(results[2].status);
    });
  });

  describe('Phase 0: Skip', () => {
    it('skips pixel_detector results', () => {
      const result = sanitizer.sanitizeWord('X', 90, 'pixel_detector');
      expect(result.status).toBe(SanitizeStatus.SKIPPED);
    });

    it('skips low-confidence single characters', () => {
      const result = sanitizer.sanitizeWord('X', 40);
      expect(result.status).toBe(SanitizeStatus.SKIPPED);
    });

    it('does not skip high-confidence single characters', () => {
      const result = sanitizer.sanitizeWord('I', 80);
      expect(result.status).not.toBe(SanitizeStatus.SKIPPED);
    });
  });

  describe('Case preservation', () => {
    it('preserves uppercase', () => {
      const result = sanitizer.sanitizeWord('DECUMENT', 60);
      expect(result.sanitizedText).toBe('DOCUMENT');
    });

    it('preserves lowercase', () => {
      const result = sanitizer.sanitizeWord('decument', 60);
      expect(result.sanitizedText).toBe('document');
    });

    it('preserves title case', () => {
      const result = sanitizer.sanitizeWord('Decument', 60);
      expect(result.sanitizedText).toBe('Document');
    });
  });

  describe('Punctuation handling', () => {
    it('preserves punctuation in corrected words', () => {
      const result = sanitizer.sanitizeWord('Decument.', 60);
      expect(result.sanitizedText).toBe('Document.');
    });

    it('handles leading and trailing punctuation', () => {
      const result = sanitizer.sanitizeWord('"Decument"', 60);
      expect(result.sanitizedText).toBe('"Document"');
    });
  });

  describe('Edge cases', () => {
    it('handles empty string', () => {
      const result = sanitizer.sanitizeWord('', 60);
      expect(result.sanitizedText).toBe('');
    });

    it('handles single punctuation', () => {
      const result = sanitizer.sanitizeWord('.', 60);
      expect(result.sanitizedText).toBe('.');
    });

    it('returns uncertain for non-dictionary words below threshold', () => {
      const result = sanitizer.sanitizeWord('XYZQWERTY', 60);
      expect(result.status).toBe(SanitizeStatus.UNCERTAIN);
    });
  });

  describe('Performance (AC7)', () => {
    it('processes 100 words in under 100ms', () => {
      // Generate 100 words with mix of valid, misspelled, and random
      const testWords = [
        // Valid words
        'Document', 'Patient', 'Information', 'Veterinary', 'Associates',
        'Compassionate', 'Care', 'Feline', 'Friends', 'Northwest',
        // Misspelled words
        'Decument', 'Compassianae', 'Speties', 'Domeelic', 'Folinn',
        // Numbers and dates
        '555-0123', '01/15/2025', '97205', 'test@example.com', 'INK-2025-',
        // More valid words
        'Medical', 'History', 'Weight', 'Breed', 'Species',
        'Owner', 'Phone', 'Address', 'Email', 'Emergency',
      ];

      // Expand to 100 words by cycling
      const words: Array<{ text: string; confidence: number; engine?: string }> = [];
      for (let i = 0; i < 100; i++) {
        words.push({
          text: testWords[i % testWords.length],
          confidence: 70 + (i % 30),
        });
      }

      const start = performance.now();
      const results = sanitizer.sanitizeWords(words);
      const elapsed = performance.now() - start;

      // Verify all words processed
      expect(results.length).toBe(100);

      // AC7: <100ms for 100 words
      expect(elapsed).toBeLessThan(100);

      console.log(`Performance: 100 words in ${elapsed.toFixed(2)}ms (${(elapsed / 100).toFixed(2)}ms/word)`);
    });
  });
});
