/**
 * Tests for multi-signal ranking module.
 * Port of tests/test_sanitize_ranking.py
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  FrequencyRanker,
  DocumentRanker,
  BigramRanker,
  OCRErrorModel,
  MultiSignalRanker,
} from './ranking.js';

describe('FrequencyRanker', () => {
  it('should load frequency data', async () => {
    const ranker = new FrequencyRanker({
      enabled: true,
      weight: 1.0,
      source: '/public/data/frequencies.json',
      fallbackFrequency: 1,
    });

    await ranker.load();

    // Check that common words have higher factors
    const fileFreq = ranker.getFrequencyFactor('file');
    const fielFreq = ranker.getFrequencyFactor('fiel');

    expect(fileFreq).toBeGreaterThan(fielFreq);
  });

  it('should rank common words higher than rare words', async () => {
    const ranker = new FrequencyRanker({
      enabled: true,
      weight: 1.0,
      source: '/public/data/frequencies.json',
      fallbackFrequency: 1,
    });

    await ranker.load();

    // "file" is common, "fiel" is rare/nonexistent
    const fileFactor = ranker.getFrequencyFactor('file');
    const fielFactor = ranker.getFrequencyFactor('fiel');

    expect(fileFactor).toBeGreaterThan(fielFactor);
    expect(fileFactor).toBeGreaterThan(0.5); // Common word should have high factor
  });

  it('should use fallback frequency for unknown words', async () => {
    const ranker = new FrequencyRanker({
      enabled: true,
      weight: 1.0,
      source: '/public/data/frequencies.json',
      fallbackFrequency: 1,
    });

    await ranker.load();

    const unknownFactor = ranker.getFrequencyFactor('xyzabc123');
    expect(unknownFactor).toBeGreaterThan(0);
    expect(unknownFactor).toBeLessThan(0.1); // Should be very low
  });

  it('should return 1.0 when disabled', () => {
    const ranker = new FrequencyRanker({
      enabled: false,
      weight: 1.0,
      source: '/public/data/frequencies.json',
      fallbackFrequency: 1,
    });

    const factor = ranker.getFrequencyFactor('file');
    expect(factor).toBe(1.0);
  });
});

describe('DocumentRanker', () => {
  it('should build document index', () => {
    const ranker = new DocumentRanker({
      enabled: true,
      weight: 0.3,
      minOccurrences: 2,
    });

    ranker.buildDocumentIndex(['file', 'File', 'edit', 'file']);

    // "file" appears 3 times (case-insensitive)
    const fileFactor = ranker.getDocumentFactor('file');
    expect(fileFactor).toBeGreaterThan(1.0);

    // "edit" appears only once
    const editFactor = ranker.getDocumentFactor('edit');
    expect(editFactor).toBe(1.0); // Below threshold
  });

  it('should boost words appearing twice or more', () => {
    const ranker = new DocumentRanker({
      enabled: true,
      weight: 0.3,
      minOccurrences: 2,
    });

    ranker.buildDocumentIndex(['file', 'file', 'edit']);

    const fileFactor = ranker.getDocumentFactor('file');
    const editFactor = ranker.getDocumentFactor('edit');

    expect(fileFactor).toBeGreaterThan(1.0);
    expect(editFactor).toBe(1.0);
  });

  it('should not boost words appearing once', () => {
    const ranker = new DocumentRanker({
      enabled: true,
      weight: 0.3,
      minOccurrences: 2,
    });

    ranker.buildDocumentIndex(['file', 'edit', 'view']);

    const fileFactor = ranker.getDocumentFactor('file');
    expect(fileFactor).toBe(1.0); // Only 1 occurrence
  });

  it('should be case-insensitive', () => {
    const ranker = new DocumentRanker({
      enabled: true,
      weight: 0.3,
      minOccurrences: 2,
    });

    ranker.buildDocumentIndex(['File', 'FILE', 'file']);

    const factor1 = ranker.getDocumentFactor('file');
    const factor2 = ranker.getDocumentFactor('File');
    const factor3 = ranker.getDocumentFactor('FILE');

    expect(factor1).toBeGreaterThan(1.0);
    expect(factor1).toBe(factor2);
    expect(factor2).toBe(factor3);
  });

  it('should return 1.0 when disabled', () => {
    const ranker = new DocumentRanker({
      enabled: false,
      weight: 0.3,
      minOccurrences: 2,
    });

    ranker.buildDocumentIndex(['file', 'file', 'file']);

    const factor = ranker.getDocumentFactor('file');
    expect(factor).toBe(1.0);
  });
});

describe('BigramRanker', () => {
  it('should load bigram data', async () => {
    const ranker = new BigramRanker({
      enabled: true,
      weight: 0.5,
      source: '/public/data/bigrams.json',
      window: 1,
    });

    await ranker.load();

    // Common bigram "of the" should have reasonable factor
    const factor = ranker.getBigramFactor('the', 'of', undefined);
    expect(factor).toBeGreaterThan(0.5);
  });

  it('should score common bigrams higher', async () => {
    const ranker = new BigramRanker({
      enabled: true,
      weight: 0.5,
      source: '/public/data/bigrams.json',
      window: 1,
    });

    await ranker.load();

    // "of the" is very common
    const commonFactor = ranker.getBigramFactor('the', 'of', undefined);

    // "xyz abc" is nonexistent
    const rareFactor = ranker.getBigramFactor('abc', 'xyz', undefined);

    expect(commonFactor).toBeGreaterThan(rareFactor);
  });

  it('should return neutral for unknown bigrams', async () => {
    const ranker = new BigramRanker({
      enabled: true,
      weight: 0.5,
      source: '/public/data/bigrams.json',
      window: 1,
    });

    await ranker.load();

    const factor = ranker.getBigramFactor('xyzabc', 'qwerty', undefined);
    expect(factor).toBe(1.0);
  });

  it('should handle prev word only', async () => {
    const ranker = new BigramRanker({
      enabled: true,
      weight: 0.5,
      source: '/public/data/bigrams.json',
      window: 1,
    });

    await ranker.load();

    const factor = ranker.getBigramFactor('the', 'of', undefined);
    expect(factor).toBeGreaterThan(0.5);
  });

  it('should handle next word only', async () => {
    const ranker = new BigramRanker({
      enabled: true,
      weight: 0.5,
      source: '/public/data/bigrams.json',
      window: 1,
    });

    await ranker.load();

    const factor = ranker.getBigramFactor('of', undefined, 'the');
    expect(factor).toBeGreaterThan(0.5);
  });

  it('should return 1.0 when disabled', () => {
    const ranker = new BigramRanker({
      enabled: false,
      weight: 0.5,
      source: '/public/data/bigrams.json',
      window: 1,
    });

    const factor = ranker.getBigramFactor('the', 'of', undefined);
    expect(factor).toBe(1.0);
  });
});

describe('OCRErrorModel', () => {
  it('should load OCR confusion data', async () => {
    const model = new OCRErrorModel({
      enabled: true,
      weight: 0.4,
      source: '/public/data/ocr_confusions.json',
    });

    await model.load();

    // Check that known confusion is recognized
    const factor = model.getOcrFactor('1', 'l');
    expect(factor).toBeGreaterThan(1.0);
  });

  it('should recognize l-to-1 confusion', async () => {
    const model = new OCRErrorModel({
      enabled: true,
      weight: 0.4,
      source: '/public/data/ocr_confusions.json',
    });

    await model.load();

    const factor = model.getOcrFactor('Filel', 'File1');
    expect(factor).toBeGreaterThan(1.0);
  });

  it('should recognize 0-to-O confusion', async () => {
    const model = new OCRErrorModel({
      enabled: true,
      weight: 0.4,
      source: '/public/data/ocr_confusions.json',
    });

    await model.load();

    const factor = model.getOcrFactor('0wner', 'Owner');
    expect(factor).toBeGreaterThan(1.0);
  });

  it('should recognize rn-to-m confusion', async () => {
    const model = new OCRErrorModel({
      enabled: true,
      weight: 0.4,
      source: '/public/data/ocr_confusions.json',
    });

    await model.load();

    const factor = model.getOcrFactor('kernal', 'kernel');
    expect(factor).toBeGreaterThan(1.0);
  });

  it('should return neutral for unknown edits', async () => {
    const model = new OCRErrorModel({
      enabled: true,
      weight: 0.4,
      source: '/public/data/ocr_confusions.json',
    });

    await model.load();

    const factor = model.getOcrFactor('abc', 'xyz');
    expect(factor).toBe(1.0);
  });

  it('should return 1.0 when disabled', () => {
    const model = new OCRErrorModel({
      enabled: false,
      weight: 0.4,
      source: '/public/data/ocr_confusions.json',
    });

    const factor = model.getOcrFactor('Filel', 'File1');
    expect(factor).toBe(1.0);
  });
});

describe('MultiSignalRanker Integration', () => {
  let ranker: MultiSignalRanker;

  beforeAll(async () => {
    ranker = new MultiSignalRanker(
      {
        enabled: true,
        weight: 1.0,
        source: '/public/data/frequencies.json',
        fallbackFrequency: 1,
      },
      {
        enabled: true,
        weight: 0.3,
        minOccurrences: 2,
      },
      {
        enabled: true,
        weight: 0.5,
        source: '/public/data/bigrams.json',
        window: 1,
      },
      {
        enabled: true,
        weight: 0.4,
        source: '/public/data/ocr_confusions.json',
      }
    );

    await ranker.load();
  });

  it('should combine all four signals', () => {
    ranker.buildDocumentIndex(['file', 'file', 'edit']);

    const [freq, doc, bigram, ocr] = ranker.rankCandidate(
      'Filel',
      'File',
      undefined,
      'edit'
    );

    expect(freq).toBeGreaterThan(0.5); // Common word
    expect(doc).toBeGreaterThan(1.0);  // Appears twice
    expect(bigram).toBeGreaterThan(0); // Has context
    expect(ocr).toBeGreaterThan(1.0);  // Known OCR error (l->e)

    const composite = ranker.getCompositeFactor('Filel', 'File', undefined, 'edit');
    expect(composite).toBeGreaterThan(1.0);
  });

  it('should correctly rank "Filel" -> "File" over "Fiel"', () => {
    ranker.buildDocumentIndex(['file', 'edit', 'view']);

    // "File" should win due to frequency
    const [freqFile, docFile, bigramFile, ocrFile] = ranker.rankCandidate(
      'Filel',
      'File',
      undefined,
      undefined
    );
    const scoreFile = freqFile * docFile * bigramFile * ocrFile;

    const [freqFiel, docFiel, bigramFiel, ocrFiel] = ranker.rankCandidate(
      'Filel',
      'Fiel',
      undefined,
      undefined
    );
    const scoreFiel = freqFiel * docFiel * bigramFiel * ocrFiel;

    expect(scoreFile).toBeGreaterThan(scoreFiel);
  });
});
