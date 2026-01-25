/**
 * Geometric clustering for document reading order.
 * Port of Python geometric_clustering.py
 *
 * Algorithm Overview:
 * 1. Calculate distance thresholds using Q1 * 1.5 of inter-word distances
 * 2. Build clusters using union-find based on spatial proximity
 * 3. Detect and reposition intra-cluster outliers
 * 4. Order clusters top-to-bottom, words within clusters left-to-right/top-to-bottom
 * 5. Assign sequential word_ids based on reading order
 */

import {
  Word,
  BBox,
  bboxCenterX,
  bboxCenterY,
  bboxHeight,
  horizontalGap as hGap,
  verticalGap as vGap,
  xOverlapRatio,
  yOverlapRatio,
} from './models';

// Configuration defaults (matching Python config/harmonize.yaml)
const CONFIG = {
  thresholds: {
    defaultX: 20.0,
    defaultY: 10.0,
    q1Multiplier: 1.5,
  },
  yFuzz: {
    default: 5.0,
    multiplier: 2.0,
    maxHeightRatio: 0.5,
  },
  connection: {
    xOverlapMin: 0.3,
    yOverlapMin: 0.5,
    verticalMultiplier: 2.0,
    sameRowXMultiplier: 3.0,
  },
  intraCluster: {
    outlierMultiplier: 3.0,
  },
};

// Helper to calculate horizontal gap between two words
function horizontalGapWords(w1: Word, w2: Word): number {
  return hGap(w1.bbox, w2.bbox);
}

// Helper to calculate vertical gap between two words
function verticalGapWords(w1: Word, w2: Word): number {
  return vGap(w1.bbox, w2.bbox);
}

// Percentile calculation (equivalent to numpy.percentile)
function percentile(arr: number[], p: number): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (index - lower);
}

// Standard deviation
function std(arr: number[]): number {
  if (arr.length === 0) return 0;
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  const squareDiffs = arr.map((x) => Math.pow(x - mean, 2));
  return Math.sqrt(squareDiffs.reduce((a, b) => a + b, 0) / arr.length);
}

// Median
function median(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sorted = [...arr].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Mean
function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

/**
 * Union-Find data structure for clustering.
 */
class UnionFind {
  private parent: number[];
  private rank: number[];

  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x: number): number {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // Path compression
    }
    return this.parent[x];
  }

  union(x: number, y: number): void {
    const px = this.find(x);
    const py = this.find(y);
    if (px === py) return;

    // Union by rank
    if (this.rank[px] < this.rank[py]) {
      this.parent[px] = py;
    } else if (this.rank[px] > this.rank[py]) {
      this.parent[py] = px;
    } else {
      this.parent[py] = px;
      this.rank[px]++;
    }
  }
}

/**
 * A group of spatially related words.
 */
interface Cluster {
  words: Word[];
}

function clusterBoundingBox(cluster: Cluster): BBox | null {
  if (cluster.words.length === 0) return null;
  const x0 = Math.min(...cluster.words.map((w) => w.bbox.x0));
  const y0 = Math.min(...cluster.words.map((w) => w.bbox.y0));
  const x1 = Math.max(...cluster.words.map((w) => w.bbox.x1));
  const y1 = Math.max(...cluster.words.map((w) => w.bbox.y1));
  return { x0, y0, x1, y1 };
}

function clusterMinY(cluster: Cluster): number {
  if (cluster.words.length === 0) return 0;
  return Math.min(...cluster.words.map((w) => w.bbox.y0));
}

function clusterMinX(cluster: Cluster): number {
  if (cluster.words.length === 0) return 0;
  return Math.min(...cluster.words.map((w) => w.bbox.x0));
}

/**
 * Calculate clustering thresholds using Q1 * multiplier of inter-word distances.
 */
function calculateDistanceThresholds(words: Word[]): [number, number] {
  if (words.length < 2) {
    return [CONFIG.thresholds.defaultX, CONFIG.thresholds.defaultY];
  }

  // Sort by rough reading order
  const sortedWords = [...words].sort((a, b) => {
    if (a.page !== b.page) return a.page - b.page;
    const aY = bboxCenterY(a.bbox);
    const bY = bboxCenterY(b.bbox);
    if (aY !== bY) return aY - bY;
    return bboxCenterX(a.bbox) - bboxCenterX(b.bbox);
  });

  const xDistances: number[] = [];
  const yDistances: number[] = [];

  for (let i = 0; i < sortedWords.length - 1; i++) {
    const w1 = sortedWords[i];
    const w2 = sortedWords[i + 1];

    if (w1.page !== w2.page) continue;

    const xDist = horizontalGapWords(w1, w2);
    const yDist = verticalGapWords(w1, w2);

    if (xDist > 0) xDistances.push(xDist);
    if (yDist > 0) yDistances.push(yDist);
  }

  const q1Mult = CONFIG.thresholds.q1Multiplier;

  let xThreshold = CONFIG.thresholds.defaultX;
  let yThreshold = CONFIG.thresholds.defaultY;

  if (xDistances.length > 0) {
    const xQ1 = percentile(xDistances, 25);
    xThreshold = xQ1 * q1Mult;
  }

  if (yDistances.length > 0) {
    const yQ1 = percentile(yDistances, 25);
    yThreshold = yQ1 * q1Mult;
  }

  // Ensure minimum thresholds
  xThreshold = Math.max(xThreshold, 5.0);
  yThreshold = Math.max(yThreshold, 3.0);

  return [xThreshold, yThreshold];
}

/**
 * Estimate y-fuzz tolerance based on y-variability between horizontally adjacent words.
 */
function estimateYFuzz(words: Word[], xThreshold: number): number {
  if (words.length < 2) {
    return CONFIG.yFuzz.default;
  }

  const yCenterDiffs: number[] = [];
  const adjacentXThreshold = xThreshold * 3.0;

  for (let i = 0; i < words.length; i++) {
    for (let j = i + 1; j < words.length; j++) {
      const w1 = words[i];
      const w2 = words[j];

      if (w1.page !== w2.page) continue;

      const xGap = horizontalGapWords(w1, w2);
      if (xGap > adjacentXThreshold) continue;

      const yGap = verticalGapWords(w1, w2);
      const minHeight = Math.min(bboxHeight(w1.bbox), bboxHeight(w2.bbox));

      if (yGap > minHeight * 0.5) continue;

      const yDiff = Math.abs(bboxCenterY(w1.bbox) - bboxCenterY(w2.bbox));
      yCenterDiffs.push(yDiff);
    }
  }

  if (yCenterDiffs.length < 3) {
    return CONFIG.yFuzz.default;
  }

  const yStd = std(yCenterDiffs);
  const yMedian = median(yCenterDiffs);

  let yFuzz = Math.max(
    yMedian + yStd,
    yStd * CONFIG.yFuzz.multiplier,
    CONFIG.yFuzz.default
  );

  // Cap at fraction of average line height
  const avgHeight = mean(words.map((w) => bboxHeight(w.bbox)));
  yFuzz = Math.min(yFuzz, avgHeight * CONFIG.yFuzz.maxHeightRatio);

  return yFuzz;
}

/**
 * Detect major vertical column boundaries in the document layout.
 */
function detectColumnBoundaries(words: Word[], yFuzz: number): number[] {
  if (words.length < 2) return [];

  // Group words into rows based on y-center proximity
  const sortedByY = [...words].sort((a, b) => bboxCenterY(a.bbox) - bboxCenterY(b.bbox));

  const rows: Word[][] = [];
  let currentRow: Word[] = [sortedByY[0]];
  let rowYCenterMax = bboxCenterY(sortedByY[0].bbox);

  for (let i = 1; i < sortedByY.length; i++) {
    const word = sortedByY[i];
    const wordYCenter = bboxCenterY(word.bbox);

    if (wordYCenter <= rowYCenterMax + yFuzz) {
      currentRow.push(word);
      rowYCenterMax = Math.max(rowYCenterMax, wordYCenter);
    } else {
      rows.push(currentRow);
      currentRow = [word];
      rowYCenterMax = wordYCenter;
    }
  }
  rows.push(currentRow);

  // Find gaps within each row
  const allGaps: [number, number][] = []; // [gap size, position]

  for (const row of rows) {
    if (row.length < 2) continue;
    const sortedRow = [...row].sort((a, b) => a.bbox.x0 - b.bbox.x0);

    for (let i = 0; i < sortedRow.length - 1; i++) {
      const w1 = sortedRow[i];
      const w2 = sortedRow[i + 1];
      const gap = w2.bbox.x0 - w1.bbox.x1;

      if (gap > 0) {
        allGaps.push([gap, (w1.bbox.x1 + w2.bbox.x0) / 2]);
      }
    }
  }

  if (allGaps.length === 0) return [];

  const gapSizes = allGaps.map((g) => g[0]);
  const medianGap = median(gapSizes);
  const threshold = medianGap * 3;

  const significantGaps = allGaps.filter(([size]) => size >= threshold);
  if (significantGaps.length === 0) return [];

  const boundaries = [...new Set(significantGaps.map(([, pos]) => pos))].sort((a, b) => a - b);
  return boundaries;
}

/**
 * Assign a word to a column based on its position relative to boundaries.
 */
function assignColumn(word: Word, boundaries: number[]): number {
  const wordCenterX = bboxCenterX(word.bbox);
  for (let i = 0; i < boundaries.length; i++) {
    if (wordCenterX < boundaries[i]) return i;
  }
  return boundaries.length;
}

/**
 * Build clusters using union-find based on spatial proximity and column detection.
 */
function buildClusters(
  words: Word[],
  xThreshold: number,
  yThreshold: number,
  yFuzz: number
): Cluster[] {
  if (words.length === 0) return [];

  const n = words.length;
  const uf = new UnionFind(n);

  // Detect column boundaries
  const boundaries = detectColumnBoundaries(words, yFuzz);

  // Assign words to columns
  const wordColumns = words.map((w) => assignColumn(w, boundaries));

  const { xOverlapMin, yOverlapMin, verticalMultiplier, sameRowXMultiplier } = CONFIG.connection;

  // Check all pairs and union if they should be connected
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const w1 = words[i];
      const w2 = words[j];

      if (w1.page !== w2.page) continue;

      const xDist = horizontalGapWords(w1, w2);
      const yDist = verticalGapWords(w1, w2);
      const xOverlap = xOverlapRatio(w1.bbox, w2.bbox);
      const yOverlap = yOverlapRatio(w1.bbox, w2.bbox);

      let shouldConnect = false;

      // Rule 1: Same row (y-overlap) AND horizontally close
      if (yOverlap >= yOverlapMin && xDist <= xThreshold * sameRowXMultiplier) {
        const isSmallGap = xDist <= xThreshold * 2.0;
        const sameColumn = wordColumns[i] === wordColumns[j];

        if (sameColumn || isSmallGap) {
          shouldConnect = true;
        }
      }
      // Rules 2 and 3: respect column boundaries
      else if (wordColumns[i] === wordColumns[j]) {
        // Rule 2: Column aligned AND vertically close
        if (xOverlap >= xOverlapMin && yDist <= yThreshold * verticalMultiplier) {
          shouldConnect = true;
        }
        // Rule 3: Very close in both dimensions
        else if (xDist <= xThreshold && yDist <= yThreshold) {
          shouldConnect = true;
        }
      }

      if (shouldConnect) {
        uf.union(i, j);
      }
    }
  }

  // Group words by cluster root
  const clustersMap = new Map<number, Word[]>();
  for (let i = 0; i < n; i++) {
    const root = uf.find(i);
    if (!clustersMap.has(root)) {
      clustersMap.set(root, []);
    }
    clustersMap.get(root)!.push(words[i]);
  }

  return Array.from(clustersMap.values()).map((words) => ({ words }));
}

/**
 * Sort words within a cluster by reading order.
 */
function sortWordsWithinCluster(cluster: Cluster, yFuzz: number): Word[] {
  if (cluster.words.length === 0) return [];

  const words = [...cluster.words];

  // Group into rows based on y-center proximity
  const sortedByY = words.sort((a, b) => bboxCenterY(a.bbox) - bboxCenterY(b.bbox));

  const rows: Word[][] = [];
  let currentRow: Word[] = [sortedByY[0]];
  let rowYCenterMax = bboxCenterY(sortedByY[0].bbox);

  for (let i = 1; i < sortedByY.length; i++) {
    const word = sortedByY[i];
    const wordYCenter = bboxCenterY(word.bbox);

    if (wordYCenter <= rowYCenterMax + yFuzz) {
      currentRow.push(word);
      rowYCenterMax = Math.max(rowYCenterMax, wordYCenter);
    } else {
      rows.push(currentRow);
      currentRow = [word];
      rowYCenterMax = wordYCenter;
    }
  }
  rows.push(currentRow);

  // Sort each row left-to-right, then flatten
  const result: Word[] = [];
  for (const row of rows) {
    const rowSorted = [...row].sort((a, b) => a.bbox.x0 - b.bbox.x0);
    result.push(...rowSorted);
  }

  return result;
}

/**
 * Group clusters into row bands based on y-overlap.
 */
function groupClustersIntoRowBands(clusters: Cluster[], yTolerance: number): Cluster[][] {
  if (clusters.length === 0) return [];

  // Sort clusters by min_y first
  const sortedClusters = [...clusters].sort((a, b) => clusterMinY(a) - clusterMinY(b));

  const rowBands: Cluster[][] = [];
  let currentBand: Cluster[] = [sortedClusters[0]];
  let bandYMax = clusterBoundingBox(sortedClusters[0])?.y1 ?? clusterMinY(sortedClusters[0]);

  for (let i = 1; i < sortedClusters.length; i++) {
    const cluster = sortedClusters[i];
    const clusterYMin = clusterMinY(cluster);

    if (clusterYMin <= bandYMax + yTolerance) {
      currentBand.push(cluster);
      const bbox = clusterBoundingBox(cluster);
      if (bbox) {
        bandYMax = Math.max(bandYMax, bbox.y1);
      }
    } else {
      rowBands.push(currentBand);
      currentBand = [cluster];
      bandYMax = clusterBoundingBox(cluster)?.y1 ?? clusterMinY(cluster);
    }
  }
  rowBands.push(currentBand);

  return rowBands;
}

/**
 * Order words by proper reading sequence using geometric clustering.
 * Main entry point for the reading order algorithm.
 */
export function orderWordsByReading(words: Word[]): Word[] {
  if (words.length === 0) return [];

  // Group by page first
  const pages = new Map<number, Word[]>();
  for (const word of words) {
    if (!pages.has(word.page)) {
      pages.set(word.page, []);
    }
    pages.get(word.page)!.push(word);
  }

  const result: Word[] = [];

  // Process pages in order
  const sortedPageNums = [...pages.keys()].sort((a, b) => a - b);

  for (const pageNum of sortedPageNums) {
    const pageWords = pages.get(pageNum)!;

    // Calculate thresholds for this page
    const [xThresh, yThresh] = calculateDistanceThresholds(pageWords);

    // Estimate y-fuzz for row grouping
    const yFuzz = estimateYFuzz(pageWords, xThresh);

    // Build clusters
    const clusters = buildClusters(pageWords, xThresh, yThresh, yFuzz);

    // Group clusters into row bands
    const rowBands = groupClustersIntoRowBands(clusters, yFuzz);

    // Process each row band: sort clusters left-to-right within band
    for (const band of rowBands) {
      const bandSorted = [...band].sort((a, b) => clusterMinX(a) - clusterMinX(b));

      for (const cluster of bandSorted) {
        const orderedClusterWords = sortWordsWithinCluster(cluster, yFuzz);
        result.push(...orderedClusterWords);
      }
    }
  }

  // Assign sequential word_ids
  for (let i = 0; i < result.length; i++) {
    result[i] = { ...result[i], wordId: i };
  }

  return result;
}
