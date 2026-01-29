/**
 * Browser smoke test using Playwright.
 * Verifies real fetch() calls work in browser environment.
 *
 * Uses test-ranking.html which properly loads the ranking module via Vite.
 */

import { test, expect } from '@playwright/test';

// Extend window type for test helpers
declare global {
  interface Window {
    testReady: boolean;
    testHelpers: {
      testFrequencyLoad(): Promise<{ factor: number; loaded: boolean }>;
      testOCRLoad(): Promise<{ factor: number; loaded: boolean }>;
      testFileRanking(): Promise<{ scoreFile: number; scoreFiel: number; fileWins: boolean }>;
    };
  }
}

test.describe('Browser Smoke Test', () => {
  test.beforeEach(async ({ page }) => {
    // Navigate to the test page that loads ranking module via Vite
    await page.goto('/test-ranking.html');

    // Wait for the module to be ready
    await page.waitForFunction(() => window.testReady === true, { timeout: 10000 });
  });

  test('loads frequency data via fetch', async ({ page }) => {
    const result = await page.evaluate(() => window.testHelpers.testFrequencyLoad());

    expect(result.loaded).toBe(true);
    expect(result.factor).toBeGreaterThan(0.5);
  });

  test('loads OCR confusions via fetch', async ({ page }) => {
    const result = await page.evaluate(() => window.testHelpers.testOCRLoad());

    expect(result.loaded).toBe(true);
    expect(result.factor).toBeGreaterThan(1.0);
  });

  test('Filel -> File ranks higher than Fiel in browser', async ({ page }) => {
    const result = await page.evaluate(() => window.testHelpers.testFileRanking());

    expect(result.fileWins).toBe(true);
    expect(result.scoreFile).toBeGreaterThan(result.scoreFiel);
  });
});
