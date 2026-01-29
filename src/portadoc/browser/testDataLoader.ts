/**
 * Test data loader for ranking tests.
 * Loads JSON files directly from filesystem (Node.js only).
 */

import * as fs from 'fs';
import * as path from 'path';

export interface Confusion {
  pattern: string;
  confused_with: string[];
  probability: number;
  context?: 'end' | 'middle';
}

export function loadFrequencies(): Record<string, number> {
  const filePath = path.join(__dirname, 'public', 'data', 'frequencies.json');
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

export function loadBigrams(): Record<string, number> {
  const filePath = path.join(__dirname, 'public', 'data', 'bigrams.json');
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}

export function loadOcrConfusions(): { confusions: Confusion[] } {
  const filePath = path.join(__dirname, 'public', 'data', 'ocr_confusions.json');
  return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
}
