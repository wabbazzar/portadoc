/**
 * Geometric clustering for document reading order.
 * Port of Python geometric_clustering.py
 *
 * Algorithm Overview:
 * 1. Calculate distance thresholds using Q1 * 1.5 of inter-word distances
 * 2. Detect significant horizontal gaps ("block gaps") that may separate content blocks
 * 3. Build clusters using union-find based on spatial proximity and block boundaries
 * 4. Order clusters by row bands (top-to-bottom), then left-to-right within each band
 * 5. Within each cluster, order words by rows (top-to-bottom, left-to-right)
 * 6. Assign sequential word_ids based on reading order
 *
 * Note: "Block gaps" are significant horizontal gaps detected within rows.
 * They may indicate column boundaries, but also table cells, side-by-side blocks, etc.
 * The term "block gap" is more accurate than "column" since gaps don't necessarily
 * span the full page height.
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

/**
 * Metadata about clustering decisions for a single cluster.
 * Useful for debugging and visualization.
 */
export interface ClusterInfo {
  id: number;
  words: Word[];
  bbox: BBox;
  wordCount: number;
  blockIndex: number; // Which block gap region this cluster belongs to
  bandIndex: number; // Which row band this cluster belongs to
  // Decision metadata
  metadata: {
    xThreshold: number;
    yThreshold: number;
    yFuzz: number;
    connectionReasons: string[]; // Why words were grouped (e.g., "same-row", "vertical-stack")
  };
}

/**
 * Result of the ordering algorithm with cluster information.
 */
export interface OrderingResult {
  words: Word[];
  clusters: ClusterInfo[];
  blockGaps: number[]; // X-positions of detected block gaps
  thresholds: {
    xThreshold: number;
    yThreshold: number;
    yFuzz: number;
  };
}

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
 * Internal cluster representation during algorithm execution.
 */
interface Cluster {
  words: Word[];
  connectionReasons: Set<string>; // Track why words were connected
}

function clusterBoundingBox(cluster: Cluster): BBox {
  if (cluster.words.length === 0) {
    return { x0: 0, y0: 0, x1: 0, y1: 0 };
  }
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
 * Detect significant horizontal gaps ("block gaps") in the document layout.
 *
 * These gaps may indicate:
 * - Column boundaries (newspaper-style layout)
 * - Side-by-side content blocks (address | contact info)
 * - Table cell separations
 * - Any significant horizontal separation
 *
 * The term "block gap" is preferred over "column" because these gaps
 * don't necessarily span the full page height.
 */
function detectBlockGaps(words: Word[], yFuzz: number): number[] {
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
 * Assign a word to a block region based on its position relative to gap boundaries.
 */
function assignBlockIndex(word: Word, blockGaps: number[]): number {
  const wordCenterX = bboxCenterX(word.bbox);
  for (let i = 0; i < blockGaps.length; i++) {
    if (wordCenterX < blockGaps[i]) return i;
  }
  return blockGaps.length;
}

/**
 * Result of building clusters, including detected block gaps.
 */
interface BuildClustersResult {
  clusters: Cluster[];
  blockGaps: number[];
}

/**
 * Build clusters using union-find based on spatial proximity and block gap detection.
 *
 * Clustering rules:
 * 1. Same row (y-overlap ≥ 50%) AND horizontally close (≤ 3x threshold)
 *    - Small gaps (≤ 2x threshold) ignore block boundaries (handles headers like "NORTHWEST VETERINARY")
 *    - Large gaps respect block boundaries (keeps "Grove" and "Email:" separate)
 * 2. Vertically stacked (x-overlap ≥ 30%) AND vertically close (≤ 2x threshold)
 *    - Only within same block region
 * 3. Very close in both dimensions (fallback)
 *    - Only within same block region
 */
function buildClusters(
  words: Word[],
  xThreshold: number,
  yThreshold: number,
  yFuzz: number
): BuildClustersResult {
  if (words.length === 0) return { clusters: [], blockGaps: [] };

  const n = words.length;
  const uf = new UnionFind(n);

  // Detect block gap boundaries
  const blockGaps = detectBlockGaps(words, yFuzz);

  // Assign words to block regions
  const wordBlocks = words.map((w) => assignBlockIndex(w, blockGaps));

  // Track connection reasons for each word pair
  const connectionReasons = new Map<string, string>(); // "i,j" -> reason

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
      let reason = '';

      // Rule 1: Same row (y-overlap) AND horizontally close
      if (yOverlap >= yOverlapMin && xDist <= xThreshold * sameRowXMultiplier) {
        const isSmallGap = xDist <= xThreshold * 2.0;
        const sameBlock = wordBlocks[i] === wordBlocks[j];

        if (sameBlock || isSmallGap) {
          shouldConnect = true;
          reason = isSmallGap ? 'same-row-close' : 'same-row-same-block';
        }
      }
      // Rules 2 and 3: respect block boundaries
      else if (wordBlocks[i] === wordBlocks[j]) {
        // Rule 2: Vertically stacked (x-overlap) AND vertically close
        if (xOverlap >= xOverlapMin && yDist <= yThreshold * verticalMultiplier) {
          shouldConnect = true;
          reason = 'vertical-stack';
        }
        // Rule 3: Very close in both dimensions
        else if (xDist <= xThreshold && yDist <= yThreshold) {
          shouldConnect = true;
          reason = 'proximity';
        }
      }

      if (shouldConnect) {
        uf.union(i, j);
        connectionReasons.set(`${i},${j}`, reason);
      }
    }
  }

  // Group words by cluster root and collect connection reasons
  const clustersMap = new Map<number, { words: Word[]; reasons: Set<string> }>();
  for (let i = 0; i < n; i++) {
    const root = uf.find(i);
    if (!clustersMap.has(root)) {
      clustersMap.set(root, { words: [], reasons: new Set() });
    }
    clustersMap.get(root)!.words.push(words[i]);
  }

  // Collect reasons for each cluster
  for (const [key, reason] of connectionReasons) {
    const [i] = key.split(',').map(Number);
    const root = uf.find(i);
    clustersMap.get(root)!.reasons.add(reason);
  }

  const clusters = Array.from(clustersMap.values()).map(({ words, reasons }) => ({
    words,
    connectionReasons: reasons,
  }));

  return { clusters, blockGaps };
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
 * Human-readable descriptions for connection reasons.
 */
const REASON_DESCRIPTIONS: Record<string, string> = {
  'same-row-close': 'Words on same row with small horizontal gap',
  'same-row-same-block': 'Words on same row within same block region',
  'vertical-stack': 'Words vertically stacked with x-overlap',
  'proximity': 'Words very close in both dimensions',
};

/**
 * Order words by proper reading sequence using geometric clustering.
 * Returns full result with cluster information for visualization.
 */
export function orderWordsByReadingWithClusters(words: Word[]): OrderingResult {
  if (words.length === 0) {
    return {
      words: [],
      clusters: [],
      blockGaps: [],
      thresholds: { xThreshold: 0, yThreshold: 0, yFuzz: 0 },
    };
  }

  // Group by page first
  const pages = new Map<number, Word[]>();
  for (const word of words) {
    if (!pages.has(word.page)) {
      pages.set(word.page, []);
    }
    pages.get(word.page)!.push(word);
  }

  const resultWords: Word[] = [];
  const resultClusters: ClusterInfo[] = [];
  let allBlockGaps: number[] = [];
  let lastThresholds = { xThreshold: 0, yThreshold: 0, yFuzz: 0 };

  // Process pages in order
  const sortedPageNums = [...pages.keys()].sort((a, b) => a - b);
  let clusterIdCounter = 0;

  for (const pageNum of sortedPageNums) {
    const pageWords = pages.get(pageNum)!;

    // Calculate thresholds for this page
    const [xThresh, yThresh] = calculateDistanceThresholds(pageWords);

    // Estimate y-fuzz for row grouping
    const yFuzz = estimateYFuzz(pageWords, xThresh);

    lastThresholds = { xThreshold: xThresh, yThreshold: yThresh, yFuzz };

    // Build clusters
    const { clusters, blockGaps } = buildClusters(pageWords, xThresh, yThresh, yFuzz);
    allBlockGaps = blockGaps;

    // Group clusters into row bands
    const rowBands = groupClustersIntoRowBands(clusters, yFuzz);

    // Process each row band: sort clusters left-to-right within band
    for (let bandIdx = 0; bandIdx < rowBands.length; bandIdx++) {
      const band = rowBands[bandIdx];
      const bandSorted = [...band].sort((a, b) => clusterMinX(a) - clusterMinX(b));

      for (const cluster of bandSorted) {
        const orderedClusterWords = sortWordsWithinCluster(cluster, yFuzz);
        resultWords.push(...orderedClusterWords);

        // Determine block index from first word
        const blockIndex = cluster.words.length > 0
          ? assignBlockIndex(cluster.words[0], blockGaps)
          : 0;

        // Build cluster info
        const clusterInfo: ClusterInfo = {
          id: clusterIdCounter++,
          words: orderedClusterWords,
          bbox: clusterBoundingBox(cluster),
          wordCount: cluster.words.length,
          blockIndex,
          bandIndex: bandIdx,
          metadata: {
            xThreshold: xThresh,
            yThreshold: yThresh,
            yFuzz,
            connectionReasons: Array.from(cluster.connectionReasons).map(
              (r) => REASON_DESCRIPTIONS[r] || r
            ),
          },
        };
        resultClusters.push(clusterInfo);
      }
    }
  }

  // Assign sequential word_ids
  for (let i = 0; i < resultWords.length; i++) {
    resultWords[i] = { ...resultWords[i], wordId: i };
  }

  return {
    words: resultWords,
    clusters: resultClusters,
    blockGaps: allBlockGaps,
    thresholds: lastThresholds,
  };
}

/**
 * Order words by proper reading sequence using geometric clustering.
 * Main entry point for the reading order algorithm.
 * Returns just the ordered words (backward compatible).
 */
export function orderWordsByReading(words: Word[]): Word[] {
  return orderWordsByReadingWithClusters(words).words;
}
