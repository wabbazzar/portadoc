/**
 * Portadoc Browser - Main Application
 * Client-side PDF word extraction with OCR
 */

import { Word, ExtractedWord, BBox, HarmonizeDetails, SanitizeDetails } from './models';
import { loadPdf, renderPage, getPdfPageCount } from './pdf-loader';
import { extractWithTesseract } from './ocr/tesseract';
import { extractWithDoctr } from './ocr/doctr';
import { orderWordsByReadingWithClusters, ClusterInfo } from './geometric-clustering';
import { harmonizeWords } from './harmonize';
import { detectMissedContent, runSanityCheck, SANITY_CHECK_EXPECTED_BBOX } from './detection';
import { loadConfig } from './config';
import { Sanitizer } from './sanitize';

// State
let currentFile: File | null = null;
let pdfDocument: Awaited<ReturnType<typeof loadPdf>> | null = null;
let currentPage = 0;
let totalPages = 1;
let extractedWords: ExtractedWord[] = [];
let clusterInfo: ClusterInfo[] = [];
let orderingThresholds: { xThreshold: number; yThreshold: number; yFuzz: number } | null = null;
let blockGaps: number[] = [];
let selectedWordId: number | null = null;
let highlightedWordId: number | null = null;
let highlightedClusterId: number | null = null;
let isProcessingFile = false; // Prevent double file processing
let sanitizer: Sanitizer | null = null;
let sanitizeEnabled = false;
let canvasNaturalWidth = 0; // Store natural canvas dimensions for scaling
let canvasNaturalHeight = 0;
// Detect touch device - tooltips are disabled on touch to avoid clutter
const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

// DOM Elements
const dropZone = document.getElementById('drop-zone')!;
const dropTarget = document.getElementById('drop-target')!;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const filenameDisplay = document.getElementById('filename')!;
const emptyState = document.getElementById('empty-state')!;
const pdfViewer = document.getElementById('pdf-viewer')!;
const panelResizer = document.getElementById('panel-resizer')!;
const sidePanel = document.getElementById('side-panel')!;
const progressOverlay = document.getElementById('progress-overlay')!;
const pdfCanvas = document.getElementById('pdf-canvas') as HTMLCanvasElement;
const bboxOverlay = document.getElementById('bbox-overlay')!;
const wordsList = document.getElementById('words-list')!;
const wordCount = document.getElementById('word-count')!;
const lineCount = document.getElementById('line-count')!;
const pageInfo = document.getElementById('page-info')!;
const pageControls = document.getElementById('page-controls')!;
const prevPageBtn = document.getElementById('prev-page') as HTMLButtonElement;
const nextPageBtn = document.getElementById('next-page') as HTMLButtonElement;
const extractBtn = document.getElementById('extract-btn') as HTMLButtonElement;
const useTesseractCheckbox = document.getElementById('use-tesseract') as HTMLInputElement;
const useDoctrCheckbox = document.getElementById('use-doctr') as HTMLInputElement;
const usePixelCheckbox = document.getElementById('use-pixel') as HTMLInputElement;
const useSanitizeCheckbox = document.getElementById('use-sanitize') as HTMLInputElement;
const exportCsvBtn = document.getElementById('export-csv')!;
const exportJsonBtn = document.getElementById('export-json')!;
const progressFill = document.getElementById('progress-fill')!;
const progressText = document.getElementById('progress-text')!;
const tooltip = document.getElementById('tooltip')!;
const showTesseractCheckbox = document.getElementById('show-tesseract') as HTMLInputElement;
const showDoctrCheckbox = document.getElementById('show-doctr') as HTMLInputElement;
const showPixelCheckbox = document.getElementById('show-pixel') as HTMLInputElement;
const showClustersCheckbox = document.getElementById('show-clusters') as HTMLInputElement;
const highlightSanitizedCheckbox = document.getElementById('highlight-sanitized') as HTMLInputElement;
const highlightHarmonizedCheckbox = document.getElementById('highlight-harmonized') as HTMLInputElement;
const readingOrderContent = document.getElementById('reading-order-content')!;
const clusterTooltip = document.getElementById('cluster-tooltip')!;
const horizontalResizer = document.getElementById('horizontal-resizer')!;
const wordDetailsModal = document.getElementById('word-details-modal')!;
const wordDetailsBody = document.getElementById('word-details-body')!;
const closeDetailsBtn = document.getElementById('close-details')!;

// Initialize
async function init() {
  // Load config first
  await loadConfig();
  await initSanitizer();
  setupEventListeners();
  setupOverlayScaling();
}

/**
 * Set up responsive overlay scaling.
 * The bbox overlay needs to scale with the canvas when it's displayed smaller than its natural size.
 */
function setupOverlayScaling() {
  // Update scale on window resize
  window.addEventListener('resize', debounce(updateOverlayScale, 100));

  // Also update on orientation change (mobile)
  window.addEventListener('orientationchange', () => {
    setTimeout(updateOverlayScale, 100);
  });
}

/**
 * Calculate and apply the overlay scale to match canvas display size.
 * Called after canvas render and on window resize.
 */
function updateOverlayScale() {
  if (!pdfCanvas || canvasNaturalWidth === 0) return;

  const displayWidth = pdfCanvas.clientWidth;
  const displayHeight = pdfCanvas.clientHeight;

  // Calculate scale factor
  const scaleX = displayWidth / canvasNaturalWidth;
  const scaleY = displayHeight / canvasNaturalHeight;

  // Use the smaller scale to maintain aspect ratio (should be equal if aspect is preserved)
  const scale = Math.min(scaleX, scaleY);

  // Apply transform to overlay container
  // We set the overlay to natural size and then scale it down
  bboxOverlay.style.width = canvasNaturalWidth + 'px';
  bboxOverlay.style.height = canvasNaturalHeight + 'px';
  bboxOverlay.style.transformOrigin = 'top left';
  bboxOverlay.style.transform = `scale(${scale})`;
}

/**
 * Simple debounce function for resize events.
 */
function debounce<T extends (...args: unknown[]) => void>(fn: T, ms: number): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return function(this: unknown, ...args: Parameters<T>) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn.apply(this, args), ms);
  };
}

async function initSanitizer() {
  try {
    // Load ranking config from localStorage
    const rankingConfig = loadRankingConfig();
    sanitizer = new Sanitizer({
      skip: { pixelDetector: true, singleCharMaxConf: 50 },
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
      ranking: rankingConfig,
    });
    await sanitizer.loadDictionaries();
    console.log('Sanitizer initialized');
  } catch (err) {
    console.error('Failed to initialize sanitizer:', err);
  }
}

function loadRankingConfig() {
  const defaults = {
    frequency: { enabled: true, weight: 1.0, source: '/public/data/frequencies.json', fallbackFrequency: 1 },
    document: { enabled: true, weight: 0.3, minOccurrences: 2 },
    bigram: { enabled: true, weight: 0.5, source: '/public/data/bigrams.json', window: 1 },
    ocrModel: { enabled: true, weight: 0.4, source: '/public/data/ocr_confusions.json' },
  };

  try {
    const stored = localStorage.getItem('portadoc_ranking_config');
    if (stored) {
      return { ...defaults, ...JSON.parse(stored) };
    }
  } catch (err) {
    console.warn('Failed to load ranking config from localStorage:', err);
  }

  return defaults;
}

function saveRankingConfig() {
  const config = {
    frequency: {
      enabled: (document.getElementById('ranking-frequency') as HTMLInputElement).checked,
      weight: parseFloat((document.getElementById('ranking-frequency-weight') as HTMLInputElement).value),
      source: '/public/data/frequencies.json',
      fallbackFrequency: 1,
    },
    document: {
      enabled: (document.getElementById('ranking-document') as HTMLInputElement).checked,
      weight: parseFloat((document.getElementById('ranking-document-weight') as HTMLInputElement).value),
      minOccurrences: 2,
    },
    bigram: {
      enabled: (document.getElementById('ranking-bigram') as HTMLInputElement).checked,
      weight: parseFloat((document.getElementById('ranking-bigram-weight') as HTMLInputElement).value),
      source: '/public/data/bigrams.json',
      window: 1,
    },
    ocrModel: {
      enabled: (document.getElementById('ranking-ocr') as HTMLInputElement).checked,
      weight: parseFloat((document.getElementById('ranking-ocr-weight') as HTMLInputElement).value),
      source: '/public/data/ocr_confusions.json',
    },
  };

  try {
    localStorage.setItem('portadoc_ranking_config', JSON.stringify(config));
  } catch (err) {
    console.warn('Failed to save ranking config to localStorage:', err);
  }
}

function setupEventListeners() {
  // File upload - header button (label wraps input, so no need to call fileInput.click())
  // Only need to handle drag events and prevent click when processing
  dropZone.addEventListener('click', (e) => {
    if (isProcessingFile) {
      e.preventDefault();
      e.stopPropagation();
      console.log('Already processing a file, ignoring click');
    }
    // Don't call fileInput.click() - the label already triggers it natively
  });
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isProcessingFile) {
      dropZone.classList.add('dragover');
    }
  });
  dropZone.addEventListener('dragleave', (e) => {
    e.stopPropagation();
    dropZone.classList.remove('dragover');
  });
  dropZone.addEventListener('drop', handleFileDrop);
  fileInput.addEventListener('change', handleFileSelect);

  // File upload - empty state drop target
  dropTarget.addEventListener('click', (e) => {
    e.stopPropagation();
    if (isProcessingFile) return;
    fileInput.click();
  });
  dropTarget.addEventListener('dragover', (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isProcessingFile) {
      dropTarget.classList.add('dragover');
    }
  });
  dropTarget.addEventListener('dragleave', (e) => {
    e.stopPropagation();
    dropTarget.classList.remove('dragover');
  });
  dropTarget.addEventListener('drop', handleFileDrop);

  // Page navigation
  prevPageBtn.addEventListener('click', () => navigatePage(-1));
  nextPageBtn.addEventListener('click', () => navigatePage(1));

  // Extraction
  extractBtn.addEventListener('click', handleExtract);

  // Export
  exportCsvBtn.addEventListener('click', exportCsv);
  exportJsonBtn.addEventListener('click', exportJson);

  // Visibility toggles
  showTesseractCheckbox.addEventListener('change', () => {
    renderBboxes();
    renderWordsList();
  });
  showDoctrCheckbox.addEventListener('change', () => {
    renderBboxes();
    renderWordsList();
  });
  showPixelCheckbox.addEventListener('change', () => {
    renderBboxes();
    renderWordsList();
  });
  showClustersCheckbox.addEventListener('change', () => {
    renderBboxes();
  });
  highlightSanitizedCheckbox?.addEventListener('change', () => {
    renderBboxes();
    renderWordsList();
  });
  highlightHarmonizedCheckbox?.addEventListener('change', () => {
    renderBboxes();
    renderWordsList();
  });

  // Sanitize toggle
  if (useSanitizeCheckbox) {
    sanitizeEnabled = useSanitizeCheckbox.checked;
    useSanitizeCheckbox.addEventListener('change', () => {
      sanitizeEnabled = useSanitizeCheckbox.checked;
    });
  }

  // Ranking controls
  setupRankingControls();

  // Panel resizers
  setupPanelResizer();
  setupHorizontalResizer();
  setupMobilePanelToggle();

  // Word details modal
  closeDetailsBtn.addEventListener('click', closeWordDetails);
  wordDetailsModal.addEventListener('click', (e) => {
    if (e.target === wordDetailsModal) closeWordDetails();
  });
}

function setupRankingControls() {
  // Load saved values
  const config = loadRankingConfig();

  // Set initial values
  (document.getElementById('ranking-frequency') as HTMLInputElement).checked = config.frequency.enabled;
  (document.getElementById('ranking-frequency-weight') as HTMLInputElement).value = config.frequency.weight.toString();
  (document.getElementById('ranking-document') as HTMLInputElement).checked = config.document.enabled;
  (document.getElementById('ranking-document-weight') as HTMLInputElement).value = config.document.weight.toString();
  (document.getElementById('ranking-bigram') as HTMLInputElement).checked = config.bigram.enabled;
  (document.getElementById('ranking-bigram-weight') as HTMLInputElement).value = config.bigram.weight.toString();
  (document.getElementById('ranking-ocr') as HTMLInputElement).checked = config.ocrModel.enabled;
  (document.getElementById('ranking-ocr-weight') as HTMLInputElement).value = config.ocrModel.weight.toString();

  // Update weight displays
  updateWeightDisplay('ranking-frequency-weight');
  updateWeightDisplay('ranking-document-weight');
  updateWeightDisplay('ranking-bigram-weight');
  updateWeightDisplay('ranking-ocr-weight');

  // Add listeners
  const rankingIds = ['ranking-frequency', 'ranking-document', 'ranking-bigram', 'ranking-ocr'];
  rankingIds.forEach(id => {
    const checkbox = document.getElementById(id) as HTMLInputElement;
    const slider = document.getElementById(`${id}-weight`) as HTMLInputElement;

    checkbox?.addEventListener('change', () => {
      saveRankingConfig();
      reinitializeSanitizer();
    });

    slider?.addEventListener('input', () => {
      updateWeightDisplay(`${id}-weight`);
    });

    slider?.addEventListener('change', () => {
      saveRankingConfig();
      reinitializeSanitizer();
    });
  });
}

function updateWeightDisplay(sliderId: string) {
  const slider = document.getElementById(sliderId) as HTMLInputElement;
  const display = slider.nextElementSibling as HTMLElement;
  if (display) {
    display.textContent = parseFloat(slider.value).toFixed(1);
  }
}

async function reinitializeSanitizer() {
  await initSanitizer();
  console.log('Sanitizer reinitialized with new ranking config');
}

function handleFileDrop(e: DragEvent) {
  e.preventDefault();
  e.stopPropagation();
  dropZone.classList.remove('dragover');
  dropTarget.classList.remove('dragover');

  if (isProcessingFile) {
    console.log('Already processing a file, ignoring drop');
    return;
  }

  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    processFile(files[0]);
  }
}

function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement;
  if (input.files && input.files.length > 0) {
    const file = input.files[0];
    // Reset input so the same file can be selected again
    input.value = '';
    processFile(file);
  }
}

async function processFile(file: File) {
  // Prevent concurrent file processing
  if (isProcessingFile) {
    console.log('Already processing a file, ignoring new request');
    return;
  }

  isProcessingFile = true;
  currentFile = file;
  currentPage = 0;
  extractedWords = [];

  // Show progress immediately and wait for it to render
  showProgress(`Loading ${file.name}...`);

  // Use requestAnimationFrame to ensure overlay renders before heavy work
  await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  try {
    if (file.type === 'application/pdf') {
      updateProgress(10, 'Parsing PDF...');
      pdfDocument = await loadPdf(file);
      totalPages = getPdfPageCount(pdfDocument);
      updateProgress(50, `Loaded ${totalPages} page${totalPages > 1 ? 's' : ''}`);
    } else {
      // Image file - treat as single page
      pdfDocument = null;
      totalPages = 1;
      updateProgress(50, 'Image loaded');
    }

    // Update filename display
    filenameDisplay.textContent = file.name;

    updatePageControls();

    updateProgress(70, 'Rendering page...');
    await renderCurrentPage();

    updateProgress(100, 'Ready');

    // Brief delay to show completion state
    await new Promise((r) => setTimeout(r, 200));

    hideProgress();

    // Show viewer, hide empty state
    emptyState.classList.add('hidden');
    pdfViewer.classList.remove('hidden');
    panelResizer.classList.remove('hidden');
    sidePanel.classList.remove('hidden');
    pageControls.classList.remove('hidden');
    extractBtn.disabled = false;
  } catch (error) {
    console.error('Error loading file:', error);
    hideProgress();
    alert('Error loading file: ' + (error as Error).message);
  } finally {
    isProcessingFile = false;
  }
}

async function renderCurrentPage() {
  if (!currentFile) return;

  const ctx = pdfCanvas.getContext('2d')!;

  if (pdfDocument) {
    await renderPage(pdfDocument, currentPage, pdfCanvas);
  } else {
    // Render image file directly
    const img = new Image();
    img.src = URL.createObjectURL(currentFile);
    await new Promise<void>((resolve) => {
      img.onload = () => {
        pdfCanvas.width = img.width;
        pdfCanvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(img.src);
        resolve();
      };
    });
  }

  // Store natural canvas dimensions for overlay scaling
  canvasNaturalWidth = pdfCanvas.width;
  canvasNaturalHeight = pdfCanvas.height;

  // Re-render bboxes for current page (this sets overlay natural size)
  renderBboxes();

  // Apply responsive scaling after DOM update
  requestAnimationFrame(() => {
    updateOverlayScale();
  });
}

function navigatePage(delta: number) {
  const newPage = currentPage + delta;
  if (newPage >= 0 && newPage < totalPages) {
    currentPage = newPage;
    updatePageControls();
    renderCurrentPage();
  }
}

function updatePageControls() {
  pageInfo.textContent = `${currentPage + 1}/${totalPages}`;
  prevPageBtn.disabled = currentPage === 0;
  nextPageBtn.disabled = currentPage >= totalPages - 1;
}

async function handleExtract() {
  if (!currentFile) return;

  const useTesseract = useTesseractCheckbox.checked;
  const useDoctr = useDoctrCheckbox.checked;
  const usePixel = usePixelCheckbox.checked;

  if (!useTesseract && !useDoctr && !usePixel) {
    alert('Please select at least one extraction method');
    return;
  }

  extractBtn.disabled = true;
  showProgress('Initializing OCR...');
  extractedWords = [];

  try {
    // Get image data from canvas - Tesseract uses data URL, docTR uses ImageData
    const ctx = pdfCanvas.getContext('2d')!;
    const imageDataObj = ctx.getImageData(0, 0, pdfCanvas.width, pdfCanvas.height);
    const imageDataUrl = pdfCanvas.toDataURL('image/png');

    // DEBUG: Log canvas info
    console.log('=== EXTRACTION DEBUG ===');
    console.log('Canvas size:', pdfCanvas.width, 'x', pdfCanvas.height);
    console.log('ImageData size:', imageDataObj.width, 'x', imageDataObj.height);
    console.log('Data URL length:', imageDataUrl.length);
    console.log('First 100 chars of data URL:', imageDataUrl.substring(0, 100));

    // Check if canvas has content (not all white)
    let nonWhitePixels = 0;
    for (let i = 0; i < Math.min(imageDataObj.data.length, 10000); i += 4) {
      if (imageDataObj.data[i] < 250 || imageDataObj.data[i+1] < 250 || imageDataObj.data[i+2] < 250) {
        nonWhitePixels++;
      }
    }
    console.log('Non-white pixels in first 2500:', nonWhitePixels);
    if (nonWhitePixels < 10) {
      console.warn('WARNING: Canvas appears to be mostly white! PDF may not have rendered.');
    }

    // Run OCR engines
    let tesseractWords: Word[] = [];
    let doctrWords: Word[] = [];

    if (useTesseract) {
      updateProgress(5, 'Running Tesseract.js...');
      console.log('Starting Tesseract extraction...');
      tesseractWords = await extractWithTesseract(
        imageDataUrl,
        currentPage,
        (progress) => {
          updateProgress(progress * (useDoctr ? 50 : 100), `Tesseract: ${Math.round(progress * 100)}%`);
        }
      );
      console.log('Tesseract returned', tesseractWords.length, 'words');
      if (tesseractWords.length > 0) {
        console.log('First 5 words:', tesseractWords.slice(0, 5).map(w => w.text));
      }
    }

    if (useDoctr) {
      updateProgress(useTesseract ? 50 : 5, 'Loading docTR models...');
      const baseProgress = useTesseract ? 50 : 0;
      const progressRange = useTesseract ? 50 : 100;
      doctrWords = await extractWithDoctr(
        imageDataObj,
        currentPage,
        (progress) => {
          updateProgress(baseProgress + progress * progressRange, `docTR: ${Math.round(progress * 100)}%`);
        }
      );
    }

    // Run pixel detection if enabled
    let pixelWords: Word[] = [];
    if (usePixel) {
      updateProgress(92, 'Running pixel detection...');
      console.log('Starting pixel detection...');

      // Get existing bboxes from OCR results to avoid overlap
      const existingBboxes: BBox[] = [
        ...tesseractWords.map((w) => w.bbox),
        ...doctrWords.map((w) => w.bbox),
      ];

      // Use canvas dimensions as page dimensions (in pixels)
      // Detection is async to prevent UI freezing
      pixelWords = await detectMissedContent(
        imageDataObj,
        currentPage,
        pdfCanvas.width,
        pdfCanvas.height,
        existingBboxes
      );
      console.log('Pixel detection found', pixelWords.length, 'regions');

      // Run sanity check for peter_lou_50dpi.pdf
      if (currentFile?.name.includes('peter_lou_50dpi') && currentPage === 0) {
        const sanity = await runSanityCheck(
          imageDataObj,
          pdfCanvas.width,
          pdfCanvas.height,
          SANITY_CHECK_EXPECTED_BBOX,
          30 // tolerance in pixels
        );
        console.log('=== SANITY CHECK ===');
        console.log('Expected bbox:', SANITY_CHECK_EXPECTED_BBOX);
        console.log('Passed:', sanity.passed);
        console.log('Closest detected:', sanity.closest?.bbox);
        console.log('Distance:', sanity.distance.toFixed(1), 'px');
        console.log('Total detected:', sanity.detected.length);
        if (!sanity.passed) {
          console.warn('SANITY CHECK FAILED - pixel detection may not be working correctly');
        }
      }
    }

    // Harmonize or combine results
    updateProgress(94, 'Processing words...');
    let allWords: ExtractedWord[];

    if (useTesseract && useDoctr) {
      // Both engines - use harmonization
      allWords = harmonizeWords(tesseractWords, doctrWords, 'tesseract');
    } else if (useTesseract) {
      // Tesseract only
      allWords = tesseractWords.map((w, i) => ({
        ...w,
        wordId: i,
        sources: ['tesseract'],
      }));
    } else if (useDoctr) {
      // docTR only
      allWords = doctrWords.map((w, i) => ({
        ...w,
        wordId: i,
        sources: ['doctr'],
      }));
    } else {
      // No OCR engines - start with empty array
      allWords = [];
    }

    // Add pixel-detected regions
    if (usePixel && pixelWords.length > 0) {
      const startIdx = allWords.length;
      const pixelExtracted: ExtractedWord[] = pixelWords.map((w, i) => ({
        ...w,
        wordId: startIdx + i,
        sources: ['pixel_detector'],
      }));
      allWords = [...allWords, ...pixelExtracted];
    }

    // Order words by reading order (with cluster info for visualization)
    updateProgress(96, 'Ordering words...');
    const orderingResult = orderWordsByReadingWithClusters(allWords);

    // Store cluster information for visualization
    clusterInfo = orderingResult.clusters;
    orderingThresholds = orderingResult.thresholds;
    blockGaps = orderingResult.blockGaps;

    console.log('=== CLUSTERING DEBUG ===');
    console.log('Clusters:', clusterInfo.length);
    console.log('Block gaps:', blockGaps);
    console.log('Thresholds:', orderingThresholds);

    // Reassign word IDs after ordering
    extractedWords = orderingResult.words.map((w, i) => ({
      ...w,
      wordId: i,
    }));

    // Sanitize OCR text if enabled
    if (sanitizeEnabled && sanitizer) {
      updateProgress(97, 'Sanitizing OCR text...');
      const wordsForSanitize = extractedWords.map(w => ({
        text: w.text,
        confidence: w.confidence,
        engine: w.engine,
      }));
      const sanitizeResults = sanitizer.sanitizeWords(wordsForSanitize);

      for (let i = 0; i < extractedWords.length; i++) {
        const result = sanitizeResults[i];
        // Store sanitization details for all words (for tooltip display)
        extractedWords[i].sanitizeDetails = {
          originalText: result.originalText,
          status: result.status,
          editDistance: result.editDistance,
          correctionScore: result.correctionScore,
          matchedDictionary: result.matchedDictionary,
        };
        if (result.status === 'corrected' || result.status === 'context') {
          extractedWords[i].text = result.sanitizedText;
          extractedWords[i].sanitized = true;
        }
      }
    }

    renderBboxes();
    renderWordsList();

    // Update overlay scaling after DOM update
    requestAnimationFrame(() => {
      updateOverlayScale();
    });

    hideProgress();
  } catch (error) {
    console.error('Extraction error:', error);
    alert('Error during extraction: ' + (error as Error).message);
    hideProgress();
  } finally {
    extractBtn.disabled = false;
  }
}

function getVisibleEngines(): Set<string> {
  const engines = new Set<string>();
  if (showTesseractCheckbox.checked) engines.add('tesseract');
  if (showDoctrCheckbox.checked) engines.add('doctr');
  if (showPixelCheckbox.checked) engines.add('pixel_detector');
  return engines;
}

/**
 * Check if a word should be visible based on its sources, visible engines, and highlight modes.
 * A word is visible if:
 * - ANY of its sources is in the visible engines set, OR
 * - Sanitize highlight is on and word has sanitize details, OR
 * - Harmonize highlight is on and word is multi-source
 */
function shouldShowWord(word: ExtractedWord, visibleEngines: Set<string>): boolean {
  const sources = word.sources && word.sources.length > 0 ? word.sources : [word.engine];
  const sourceVisible = sources.some(source => visibleEngines.has(source));

  // Check if highlight modes should force visibility
  const highlightSanitized = highlightSanitizedCheckbox?.checked ?? false;
  const highlightHarmonized = highlightHarmonizedCheckbox?.checked ?? false;

  const sanitizeForceShow = highlightSanitized && word.sanitizeDetails &&
    (word.sanitizeDetails.status === 'corrected' || word.sanitizeDetails.status === 'context' || word.sanitizeDetails.status === 'uncertain');
  const harmonizeForceShow = highlightHarmonized && sources.length > 1;

  return sourceVisible || sanitizeForceShow || harmonizeForceShow;
}

function renderBboxes() {
  bboxOverlay.innerHTML = '';
  // Set overlay to natural canvas size (scaling is handled by updateOverlayScale)
  const naturalWidth = canvasNaturalWidth || pdfCanvas.width;
  const naturalHeight = canvasNaturalHeight || pdfCanvas.height;
  bboxOverlay.style.width = naturalWidth + 'px';
  bboxOverlay.style.height = naturalHeight + 'px';

  // Render cluster bboxes first (so word bboxes appear on top)
  if (showClustersCheckbox?.checked && clusterInfo.length > 0) {
    renderClusterBboxes();
  }

  const visibleEngines = getVisibleEngines();
  const pageWords = extractedWords.filter(
    (w) => w.page === currentPage && shouldShowWord(w, visibleEngines)
  );

  // Determine which source toggles are active for smart color assignment
  const showT = showTesseractCheckbox.checked;
  const showD = showDoctrCheckbox.checked;
  const highlightSanitized = highlightSanitizedCheckbox?.checked ?? false;
  const highlightHarmonized = highlightHarmonizedCheckbox?.checked ?? false;

  for (const word of pageWords) {
    const div = document.createElement('div');
    div.className = 'bbox';

    // Smart source color: only show purple (harmonized) if BOTH T and D are visible
    const sources = word.sources ?? [];
    const isMultiSource = sources.length > 1;
    const hasTesseract = sources.includes('tesseract') || word.engine === 'tesseract';
    const hasDoctr = sources.includes('doctr') || word.engine === 'doctr';
    const isPixel = word.engine === 'pixel_detector' || sources.includes('pixel_detector');

    // Determine base color class based on source visibility
    // When highlight modes force visibility, use muted styling if source toggles are off
    const sourceVisible = (hasTesseract && showT) || (hasDoctr && showD) || (isPixel && showPixelCheckbox.checked);

    if (isPixel) {
      div.classList.add('pixel_detector');
    } else if (isMultiSource && showT && showD) {
      // Both sources visible - show harmonized purple
      div.classList.add('harmonized');
    } else if (hasTesseract && showT) {
      // Tesseract visible
      div.classList.add('tesseract');
    } else if (hasDoctr && showD) {
      // docTR visible
      div.classList.add('doctr');
    } else if (!sourceVisible) {
      // Word is visible only due to highlight mode - use muted style
      div.classList.add('highlight-only');
    } else if (hasTesseract) {
      div.classList.add('tesseract');
    } else if (hasDoctr) {
      div.classList.add('doctr');
    }

    // Mark low-confidence words with yellow styling
    if (word.lowConfidence) {
      div.classList.add('low-confidence');
    }
    if (selectedWordId === word.wordId) div.classList.add('selected');
    if (highlightedWordId === word.wordId) div.classList.add('highlighted');

    // Sanitization highlight indicators
    if (highlightSanitized && word.sanitizeDetails) {
      const status = word.sanitizeDetails.status;
      if (status === 'corrected' || status === 'context') {
        div.classList.add('sanitize-corrected');
      } else if (status === 'preserved') {
        div.classList.add('sanitize-preserved');
      } else if (status === 'uncertain') {
        div.classList.add('sanitize-uncertain');
      }
    }

    // Harmonization highlight indicators
    if (highlightHarmonized && isMultiSource) {
      div.classList.add('harmonize-matched');
    }

    div.style.left = word.bbox.x0 + 'px';
    div.style.top = word.bbox.y0 + 'px';
    div.style.width = (word.bbox.x1 - word.bbox.x0) + 'px';
    div.style.height = (word.bbox.y1 - word.bbox.y0) + 'px';

    div.dataset.wordId = String(word.wordId);
    div.addEventListener('click', () => selectWord(word.wordId));
    div.addEventListener('mouseenter', (e) => highlightWord(word.wordId, e));
    div.addEventListener('mouseleave', () => unhighlightWord());

    bboxOverlay.appendChild(div);
  }
}

/**
 * Render cluster bounding boxes with color coding and tooltips.
 * Uses a color palette to distinguish different clusters.
 */
function renderClusterBboxes() {
  // Color palette for clusters (distinct, muted colors)
  const clusterColors = [
    { border: 'rgba(255, 99, 132, 0.8)', bg: 'rgba(255, 99, 132, 0.1)' },   // red
    { border: 'rgba(54, 162, 235, 0.8)', bg: 'rgba(54, 162, 235, 0.1)' },   // blue
    { border: 'rgba(255, 206, 86, 0.8)', bg: 'rgba(255, 206, 86, 0.1)' },   // yellow
    { border: 'rgba(75, 192, 192, 0.8)', bg: 'rgba(75, 192, 192, 0.1)' },   // teal
    { border: 'rgba(153, 102, 255, 0.8)', bg: 'rgba(153, 102, 255, 0.1)' }, // purple
    { border: 'rgba(255, 159, 64, 0.8)', bg: 'rgba(255, 159, 64, 0.1)' },   // orange
    { border: 'rgba(199, 199, 199, 0.8)', bg: 'rgba(199, 199, 199, 0.1)' }, // gray
    { border: 'rgba(83, 102, 255, 0.8)', bg: 'rgba(83, 102, 255, 0.1)' },   // indigo
  ];

  // Filter clusters for current page
  const pageClusters = clusterInfo.filter((c) =>
    c.words.length > 0 && c.words[0].page === currentPage
  );

  for (const cluster of pageClusters) {
    const color = clusterColors[cluster.id % clusterColors.length];
    const div = document.createElement('div');
    div.className = 'cluster-bbox';
    if (highlightedClusterId === cluster.id) {
      div.classList.add('highlighted');
    }

    div.style.left = cluster.bbox.x0 + 'px';
    div.style.top = cluster.bbox.y0 + 'px';
    div.style.width = (cluster.bbox.x1 - cluster.bbox.x0) + 'px';
    div.style.height = (cluster.bbox.y1 - cluster.bbox.y0) + 'px';
    div.style.borderColor = color.border;
    div.style.backgroundColor = color.bg;

    // Add cluster ID label
    const label = document.createElement('span');
    label.className = 'cluster-label';
    label.textContent = `C${cluster.id}`;
    label.style.backgroundColor = color.border;
    div.appendChild(label);

    div.dataset.clusterId = String(cluster.id);
    div.addEventListener('mouseenter', (e) => showClusterTooltip(cluster, e));
    div.addEventListener('mouseleave', () => hideClusterTooltip());

    bboxOverlay.appendChild(div);
  }
}

/**
 * Show tooltip with cluster details on hover (disabled on touch devices).
 */
function showClusterTooltip(cluster: ClusterInfo, event: MouseEvent) {
  if (isTouchDevice) return; // Skip tooltips on touch - no dismiss mechanism
  highlightedClusterId = cluster.id;

  // Build tooltip content
  const reasons = cluster.metadata.connectionReasons.length > 0
    ? cluster.metadata.connectionReasons.join('; ')
    : 'Single word (no connections)';

  const firstWords = cluster.words.slice(0, 3).map((w) => w.text).join(' ');
  const wordPreview = cluster.words.length > 3 ? `${firstWords}...` : firstWords;

  const html = `
    <div class="cluster-tooltip-header">Cluster ${cluster.id}</div>
    <div class="cluster-tooltip-row">
      <span class="label">Words:</span>
      <span class="value">${cluster.wordCount}</span>
    </div>
    <div class="cluster-tooltip-row">
      <span class="label">Band:</span>
      <span class="value">${cluster.bandIndex} (row group)</span>
    </div>
    <div class="cluster-tooltip-row">
      <span class="label">Block:</span>
      <span class="value">${cluster.blockIndex}</span>
    </div>
    <div class="cluster-tooltip-section">Why grouped:</div>
    <div class="cluster-tooltip-reasons">${reasons}</div>
    <div class="cluster-tooltip-section">Preview:</div>
    <div class="cluster-tooltip-preview">${escapeHtml(wordPreview)}</div>
    <div class="cluster-tooltip-thresholds">
      x-thresh: ${cluster.metadata.xThreshold.toFixed(1)}px,
      y-thresh: ${cluster.metadata.yThreshold.toFixed(1)}px,
      y-fuzz: ${cluster.metadata.yFuzz.toFixed(1)}px
    </div>
  `;

  clusterTooltip.innerHTML = html;
  clusterTooltip.style.left = (event.clientX + 15) + 'px';
  clusterTooltip.style.top = (event.clientY + 15) + 'px';
  clusterTooltip.classList.remove('hidden');

  // Highlight this cluster's bbox
  document.querySelectorAll('.cluster-bbox').forEach((el) => {
    el.classList.toggle('highlighted', el.getAttribute('data-cluster-id') === String(cluster.id));
  });
}

/**
 * Hide cluster tooltip.
 */
function hideClusterTooltip() {
  highlightedClusterId = null;
  clusterTooltip.classList.add('hidden');

  document.querySelectorAll('.cluster-bbox.highlighted').forEach((el) => {
    el.classList.remove('highlighted');
  });
}

function renderWordsList() {
  wordsList.innerHTML = '';
  const visibleEngines = getVisibleEngines();
  console.log('[renderWordsList] Total extractedWords:', extractedWords.length);
  console.log('[renderWordsList] Current page:', currentPage);
  console.log('[renderWordsList] Visible engines:', Array.from(visibleEngines));
  const pageWords = extractedWords.filter(
    (w) => w.page === currentPage && shouldShowWord(w, visibleEngines)
  );
  console.log('[renderWordsList] Filtered pageWords:', pageWords.length);
  wordCount.textContent = `${pageWords.length} words`;

  const showT = showTesseractCheckbox.checked;
  const showD = showDoctrCheckbox.checked;
  const highlightSanitized = highlightSanitizedCheckbox?.checked ?? false;
  const highlightHarmonized = highlightHarmonizedCheckbox?.checked ?? false;

  for (const word of pageWords) {
    const div = document.createElement('div');
    div.className = 'word-item';
    div.dataset.wordId = String(word.wordId);
    if (selectedWordId === word.wordId) div.classList.add('selected');
    if (highlightedWordId === word.wordId) div.classList.add('highlighted');
    if (word.lowConfidence) div.classList.add('low-confidence');

    // Add sanitization/harmonization highlight classes
    const sources = word.sources ?? [];
    const isMultiSource = sources.length > 1;

    if (highlightSanitized && word.sanitizeDetails) {
      const status = word.sanitizeDetails.status;
      if (status === 'corrected' || status === 'context') {
        div.classList.add('sanitize-corrected');
      } else if (status === 'uncertain') {
        div.classList.add('sanitize-uncertain');
      }
    }

    if (highlightHarmonized && isMultiSource) {
      div.classList.add('harmonize-matched');
    }

    const textSpan = document.createElement('span');
    textSpan.className = 'word-text';
    textSpan.textContent = word.text;

    const engineSpan = document.createElement('span');
    // Smart badge: show based on visible sources
    const hasTesseract = sources.includes('tesseract') || word.engine === 'tesseract';
    const hasDoctr = sources.includes('doctr') || word.engine === 'doctr';
    const isPixel = word.engine === 'pixel_detector' || sources.includes('pixel_detector');

    // Check if word is visible due to source toggles
    const sourceVisible = (hasTesseract && showT) || (hasDoctr && showD) || (isPixel && showPixelCheckbox.checked);

    if (isPixel) {
      engineSpan.className = 'word-engine pixel_detector';
      engineSpan.textContent = 'PX';
    } else if (isMultiSource && showT && showD) {
      engineSpan.className = 'word-engine harmonized';
      engineSpan.textContent = sources.map(s => s === 'tesseract' ? 'T' : s === 'doctr' ? 'D' : 'PX').join('');
    } else if (hasTesseract && showT) {
      engineSpan.className = 'word-engine tesseract';
      engineSpan.textContent = 'T';
    } else if (hasDoctr && showD) {
      engineSpan.className = 'word-engine doctr';
      engineSpan.textContent = 'D';
    } else if (!sourceVisible) {
      // Word visible only due to highlight mode - show muted badge
      engineSpan.className = 'word-engine muted';
      engineSpan.textContent = isMultiSource ? 'TD' : (hasTesseract ? 'T' : 'D');
    } else if (hasTesseract) {
      engineSpan.className = 'word-engine tesseract';
      engineSpan.textContent = 'T';
    } else if (hasDoctr) {
      engineSpan.className = 'word-engine doctr';
      engineSpan.textContent = 'D';
    } else {
      engineSpan.className = `word-engine ${word.engine}`;
      engineSpan.textContent = word.engine.charAt(0).toUpperCase();
    }

    div.appendChild(textSpan);
    div.appendChild(engineSpan);
    div.addEventListener('click', () => selectWord(word.wordId));
    div.addEventListener('mouseenter', () => highlightWordFromList(word.wordId));
    div.addEventListener('mouseleave', () => unhighlightWord());

    wordsList.appendChild(div);
  }

  // Also update reading order
  renderReadingOrder();
}

function selectWord(wordId: number) {
  selectedWordId = selectedWordId === wordId ? null : wordId;
  renderBboxes();
  renderWordsList();

  // Scroll word into view in list
  if (selectedWordId !== null) {
    const wordItem = wordsList.querySelector(`[data-word-id="${selectedWordId}"]`);
    wordItem?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    // Show word details modal
    const word = extractedWords.find((w) => w.wordId === wordId);
    if (word) {
      showWordDetails(word);
    }
  }
}

function highlightWord(wordId: number, event?: MouseEvent) {
  highlightedWordId = wordId;

  // Update bbox highlighting
  document.querySelectorAll('.bbox').forEach((el) => {
    el.classList.toggle('highlighted', el.getAttribute('data-word-id') === String(wordId));
  });

  // Update word list highlighting and scroll into view
  document.querySelectorAll('.word-item').forEach((el) => {
    const isHighlighted = el.getAttribute('data-word-id') === String(wordId);
    el.classList.toggle('highlighted', isHighlighted);
    if (isHighlighted) {
      el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  });

  // Show tooltip near the bbox (skip on touch devices - no dismiss mechanism)
  if (event && !isTouchDevice) {
    const word = extractedWords.find((w) => w.wordId === wordId);
    if (word && tooltip) {
      // Build tooltip content with sanitization details
      let html = `<div class="tooltip-text">${escapeHtml(word.text)}</div>`;

      if (word.sanitizeDetails) {
        const s = word.sanitizeDetails;

        if (s.status === 'corrected' || s.status === 'context') {
          html += `<div class="tooltip-sanitize corrected">`;
          html += `<span class="label">Corrected:</span> `;
          html += `<span class="original">${escapeHtml(s.originalText)}</span>`;
          html += ` → <span class="corrected">${escapeHtml(word.text)}</span>`;
          html += `</div>`;
          html += `<div class="tooltip-sanitize-details">`;
          html += `Edit distance: ${s.editDistance}`;
          if (s.matchedDictionary) {
            html += ` | Dict: ${s.matchedDictionary}`;
          }
          html += `</div>`;
          // Add ranking factors if available
          if (s.frequencyFactor || s.documentFactor || s.bigramFactor || s.ocrFactor) {
            html += `<div class="tooltip-ranking-factors" style="font-size: 0.75rem; margin-top: 4px; color: #888;">`;
            const factors = [];
            if (s.frequencyFactor !== undefined && s.frequencyFactor !== 1.0) {
              factors.push(`Freq: ${s.frequencyFactor.toFixed(2)}`);
            }
            if (s.documentFactor !== undefined && s.documentFactor !== 1.0) {
              factors.push(`Doc: ${s.documentFactor.toFixed(2)}`);
            }
            if (s.bigramFactor !== undefined && s.bigramFactor !== 1.0) {
              factors.push(`Bigram: ${s.bigramFactor.toFixed(2)}`);
            }
            if (s.ocrFactor !== undefined && s.ocrFactor !== 1.0) {
              factors.push(`OCR: ${s.ocrFactor.toFixed(2)}`);
            }
            html += factors.join(', ');
            html += `</div>`;
          }
        } else if (s.status === 'preserved') {
          html += `<div class="tooltip-sanitize preserved">`;
          html += `<span class="label">Preserved</span>`;
          if (s.matchedDictionary) {
            html += ` (${s.matchedDictionary})`;
          }
          html += `</div>`;
        } else if (s.status === 'uncertain') {
          html += `<div class="tooltip-sanitize uncertain">`;
          html += `<span class="label">Uncertain</span> (no dictionary match)`;
          html += `</div>`;
        } else if (s.status === 'skipped') {
          html += `<div class="tooltip-sanitize skipped">`;
          html += `<span class="label">Skipped</span>`;
          html += `</div>`;
        }
      }

      tooltip.innerHTML = html;
      tooltip.style.left = (event.clientX + 10) + 'px';
      tooltip.style.top = (event.clientY - 30) + 'px';
      tooltip.classList.remove('hidden');
    }
  }
}

function highlightWordFromList(wordId: number) {
  highlightedWordId = wordId;

  // Update bbox highlighting
  document.querySelectorAll('.bbox').forEach((el) => {
    el.classList.toggle('highlighted', el.getAttribute('data-word-id') === String(wordId));
  });

  // Update word list highlighting (don't scroll - user is already there)
  document.querySelectorAll('.word-item').forEach((el) => {
    el.classList.toggle('highlighted', el.getAttribute('data-word-id') === String(wordId));
  });

  // Scroll bbox into view in canvas
  const bboxEl = bboxOverlay.querySelector(`[data-word-id="${wordId}"]`) as HTMLElement;
  if (bboxEl) {
    const container = document.getElementById('pdf-container');
    if (container) {
      const bboxRect = bboxEl.getBoundingClientRect();
      const containerRect = container.getBoundingClientRect();
      // Only scroll if bbox is not visible
      if (bboxRect.top < containerRect.top || bboxRect.bottom > containerRect.bottom) {
        bboxEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }
}

function unhighlightWord() {
  highlightedWordId = null;

  // Remove highlighting from all elements
  document.querySelectorAll('.bbox.highlighted').forEach((el) => {
    el.classList.remove('highlighted');
  });
  document.querySelectorAll('.word-item.highlighted').forEach((el) => {
    el.classList.remove('highlighted');
  });

  // Hide tooltip
  if (tooltip) {
    tooltip.classList.add('hidden');
  }
}

function showProgress(text: string) {
  progressOverlay.classList.remove('hidden');
  progressText.textContent = text;
  progressFill.style.width = '0%';
}

function updateProgress(percent: number, text: string) {
  progressFill.style.width = percent + '%';
  progressText.textContent = text;
}

function hideProgress() {
  progressOverlay.classList.add('hidden');
}

function exportCsv() {
  const rows = ['page,word_id,text,x0,y0,x1,y1,engine,confidence'];
  for (const word of extractedWords) {
    const text = word.text.includes(',') ? `"${word.text}"` : word.text;
    rows.push(
      `${word.page},${word.wordId},${text},${word.bbox.x0},${word.bbox.y0},${word.bbox.x1},${word.bbox.y1},${word.engine},${word.confidence}`
    );
  }
  downloadFile(rows.join('\n'), 'portadoc-export.csv', 'text/csv');
}

function exportJson() {
  const data = {
    words: extractedWords,
    totalPages,
    exportedAt: new Date().toISOString(),
  };
  downloadFile(JSON.stringify(data, null, 2), 'portadoc-export.json', 'application/json');
}

function downloadFile(content: string, filename: string, mimeType: string) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// Panel resizer setup
function setupPanelResizer() {
  let isResizing = false;
  let startX = 0;
  let startWidth = 0;

  panelResizer.addEventListener('mousedown', (e) => {
    isResizing = true;
    startX = e.clientX;
    startWidth = sidePanel.offsetWidth;
    document.body.classList.add('resizing');
    panelResizer.classList.add('dragging');
  });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const delta = startX - e.clientX;
    const newWidth = Math.max(240, Math.min(500, startWidth + delta));
    sidePanel.style.width = newWidth + 'px';
  });

  document.addEventListener('mouseup', () => {
    if (isResizing) {
      isResizing = false;
      document.body.classList.remove('resizing');
      panelResizer.classList.remove('dragging');
    }
  });
}

// Horizontal resizer between words panel and reading order
function setupHorizontalResizer() {
  let isResizing = false;
  let startY = 0;
  let wordsPanel: HTMLElement;
  let readingPanel: HTMLElement;
  let startWordsPanelHeight = 0;

  horizontalResizer.addEventListener('mousedown', (e) => {
    isResizing = true;
    startY = e.clientY;
    wordsPanel = document.getElementById('words-panel')!;
    readingPanel = document.getElementById('reading-order-panel')!;
    startWordsPanelHeight = wordsPanel.offsetHeight;
    document.body.style.cursor = 'row-resize';
    document.body.style.userSelect = 'none';
    horizontalResizer.classList.add('dragging');
  });

  document.addEventListener('mousemove', (e) => {
    if (!isResizing) return;
    const delta = e.clientY - startY;
    const containerHeight = sidePanel.offsetHeight - horizontalResizer.offsetHeight;
    const newWordsPanelHeight = Math.max(150, Math.min(containerHeight - 150, startWordsPanelHeight + delta));
    const wordsFlex = newWordsPanelHeight / containerHeight;
    const readingFlex = 1 - wordsFlex;
    wordsPanel.style.flex = `${wordsFlex} 1 0`;
    readingPanel.style.flex = `${readingFlex} 1 0`;
  });

  document.addEventListener('mouseup', () => {
    if (isResizing) {
      isResizing = false;
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      horizontalResizer.classList.remove('dragging');
    }
  });
}

// Mobile bottom sheet panel toggle
function setupMobilePanelToggle() {
  const pullHandle = document.getElementById('panel-pull-handle');
  const wordsPanel = document.getElementById('words-panel');
  const panelHeader = wordsPanel?.querySelector('.panel-header');

  // Toggle panel expansion on pull handle click
  pullHandle?.addEventListener('click', () => {
    sidePanel.classList.toggle('expanded');
  });

  // Toggle panel expansion on header click (mobile only)
  panelHeader?.addEventListener('click', () => {
    if (window.innerWidth < 900) {
      sidePanel.classList.toggle('expanded');
    }
  });

  // Touch drag to expand/collapse (swipe gestures)
  let touchStartY = 0;
  let touchCurrentY = 0;
  let isDragging = false;

  pullHandle?.addEventListener('touchstart', (e: TouchEvent) => {
    touchStartY = e.touches[0].clientY;
    isDragging = true;
  }, { passive: true });

  document.addEventListener('touchmove', (e: TouchEvent) => {
    if (!isDragging) return;
    touchCurrentY = e.touches[0].clientY;
  }, { passive: true });

  document.addEventListener('touchend', () => {
    if (!isDragging) return;
    isDragging = false;
    const deltaY = touchCurrentY - touchStartY;
    const threshold = 50;

    if (deltaY < -threshold && !sidePanel.classList.contains('expanded')) {
      // Swiped up - expand
      sidePanel.classList.add('expanded');
    } else if (deltaY > threshold && sidePanel.classList.contains('expanded')) {
      // Swiped down - collapse
      sidePanel.classList.remove('expanded');
    }
  });

  // Close panel when clicking outside on mobile
  document.addEventListener('click', (e) => {
    if (window.innerWidth >= 900) return;
    const target = e.target as HTMLElement;
    if (!sidePanel.contains(target) && sidePanel.classList.contains('expanded')) {
      sidePanel.classList.remove('expanded');
    }
  });

  // Handle window resize
  window.addEventListener('resize', () => {
    if (window.innerWidth >= 900) {
      // Remove expanded class on desktop
      sidePanel.classList.remove('expanded');
    }
  });
}

const SOURCE_NAMES: Record<string, string> = {
  tesseract: 'Tesseract.js',
  doctr: 'docTR-TFJS',
  harmonized: 'Harmonized',
  pixel_detector: 'Pixel Detection',
};

const MATCH_TYPE_LABELS: Record<string, string> = {
  exact_text_iou: 'Exact text + bbox overlap',
  iou_only: 'Bbox overlap (text differs)',
  center_distance: 'Center proximity + exact text',
  unmatched: 'No match found',
};

function getSourceBadge(engine: string): string {
  const abbrev = engine === 'tesseract' ? 'T' : engine === 'doctr' ? 'D' : engine === 'pixel_detector' ? 'PX' : engine.charAt(0).toUpperCase();
  return `<span class="engine-badge ${engine}">${abbrev}</span>`;
}

function buildHarmonizeSection(harmonize: HarmonizeDetails): string {
  // Match type indicator
  const matchLabel = MATCH_TYPE_LABELS[harmonize.matchType] || harmonize.matchType;
  const iouDisplay = harmonize.iouScore !== undefined
    ? ` (IoU: ${(harmonize.iouScore * 100).toFixed(1)}%)`
    : '';

  // Text agreement indicator
  const agreementClass = harmonize.textAgreed ? 'agreement-yes' : 'agreement-no';
  const agreementIcon = harmonize.textAgreed ? '✓' : '✗';
  const agreementText = harmonize.textAgreed ? 'Engines agreed' : 'Engines disagreed';

  // Build contributions table
  let contributionsHtml = '';
  for (const contrib of harmonize.contributions) {
    const isPrimary = !contrib.bbox; // Primary engine doesn't have bbox in contribution
    const roleLabel = isPrimary ? '<span class="role-badge primary">bbox</span>' : '<span class="role-badge secondary">match</span>';
    contributionsHtml += `
      <tr class="contribution-row ${contrib.engine}">
        <td>${getSourceBadge(contrib.engine)} ${roleLabel}</td>
        <td class="contrib-text">${escapeHtml(contrib.text)}</td>
        <td class="contrib-conf">${contrib.confidence.toFixed(0)}%</td>
      </tr>
    `;
  }

  return `
    <div class="harmonize-section">
      <div class="harmonize-header">Harmonization Details</div>

      <div class="harmonize-match-info">
        <div class="match-type">
          <span class="match-label">Match:</span>
          <span class="match-value">${matchLabel}${iouDisplay}</span>
        </div>
        <div class="text-agreement ${agreementClass}">
          <span class="agreement-icon">${agreementIcon}</span>
          <span>${agreementText}</span>
        </div>
      </div>

      <div class="contributions-table-wrapper">
        <table class="contributions-table">
          <thead>
            <tr>
              <th>Engine</th>
              <th>Text</th>
              <th>Conf</th>
            </tr>
          </thead>
          <tbody>
            ${contributionsHtml}
          </tbody>
        </table>
      </div>

      <div class="chosen-reason">
        <span class="reason-label">Chosen because:</span>
        <span class="reason-value">${escapeHtml(harmonize.chosenTextReason)}</span>
      </div>
    </div>
  `;
}

function buildSanitizeSection(sanitize: SanitizeDetails, currentText: string): string {
  const statusLabels: Record<string, string> = {
    'skipped': 'Skipped',
    'preserved': 'Preserved',
    'corrected': 'Corrected',
    'context': 'Context Corrected',
    'uncertain': 'Uncertain',
  };

  const statusIcons: Record<string, string> = {
    'skipped': '⊘',
    'preserved': '✓',
    'corrected': '✎',
    'context': '✎',
    'uncertain': '?',
  };

  const statusLabel = statusLabels[sanitize.status] || sanitize.status;
  const statusIcon = statusIcons[sanitize.status] || '';
  const isCorrected = sanitize.status === 'corrected' || sanitize.status === 'context';

  // Correction details (only for corrected words)
  let correctionHtml = '';
  if (isCorrected) {
    correctionHtml = `
      <div class="sanitize-correction">
        <div class="correction-arrow">
          <span class="original-text">${escapeHtml(sanitize.originalText)}</span>
          <span class="arrow">→</span>
          <span class="corrected-text">${escapeHtml(currentText)}</span>
        </div>
        <div class="correction-metrics">
          <span class="metric">
            <span class="metric-label">Edit distance:</span>
            <span class="metric-value">${sanitize.editDistance}</span>
          </span>
          <span class="metric">
            <span class="metric-label">Score:</span>
            <span class="metric-value">${(sanitize.correctionScore * 100).toFixed(0)}%</span>
          </span>
        </div>
      </div>
    `;
  }

  // Dictionary match info
  let dictionaryHtml = '';
  if (sanitize.matchedDictionary) {
    const dictLabels: Record<string, string> = {
      'english': 'English Dictionary',
      'names': 'Names Dictionary',
      'medical': 'Medical Terms',
      'custom': 'Custom Dictionary',
    };
    const dictLabel = dictLabels[sanitize.matchedDictionary] || sanitize.matchedDictionary;
    dictionaryHtml = `
      <div class="sanitize-dictionary">
        <span class="dict-label">Matched:</span>
        <span class="dict-badge ${sanitize.matchedDictionary}">${dictLabel}</span>
      </div>
    `;
  }

  // Status description
  const statusDescriptions: Record<string, string> = {
    'skipped': 'Word was skipped (pixel detector or low-confidence single character)',
    'preserved': 'Word matched dictionary exactly or was high-confidence',
    'corrected': 'Word was corrected using fuzzy dictionary matching',
    'context': 'Word was corrected using surrounding context',
    'uncertain': 'No dictionary match found within edit distance threshold',
  };
  const statusDesc = statusDescriptions[sanitize.status] || '';

  return `
    <div class="sanitize-section">
      <div class="sanitize-header">Sanitization Details</div>

      <div class="sanitize-status-row">
        <div class="sanitize-status ${sanitize.status}">
          <span class="status-icon">${statusIcon}</span>
          <span class="status-label">${statusLabel}</span>
        </div>
        <div class="status-description">${statusDesc}</div>
      </div>

      ${correctionHtml}
      ${dictionaryHtml}
    </div>
  `;
}

function showWordDetails(word: ExtractedWord) {
  const sourceName = SOURCE_NAMES[word.engine] || word.engine;
  const sourceAbbrev = word.engine === 'tesseract' ? 'T' : word.engine === 'doctr' ? 'D' : word.engine === 'harmonized' ? 'TD' : word.engine === 'pixel_detector' ? 'PX' : word.engine.charAt(0).toUpperCase();
  const status = word.lowConfidence ? 'low confidence' : word.engine === 'pixel_detector' ? 'pixel' : 'text';

  // Build sources display
  let sourcesHtml = '';
  if (word.sources && word.sources.length > 0) {
    const sourceNames = word.sources.map(s => SOURCE_NAMES[s] || s).join(' + ');
    const sourceCodes = word.sources.map(s => s === 'tesseract' ? 'T' : s === 'doctr' ? 'D' : 'PX').join('');
    sourcesHtml = `
      <div class="detail-row">
        <span class="detail-label">Sources:</span>
        <span class="detail-value">
          <span class="engine-badge ${word.sources.length > 1 ? 'harmonized' : word.engine}">${sourceCodes}</span>
          ${sourceNames}
        </span>
      </div>
    `;
  } else {
    sourcesHtml = `
      <div class="detail-row">
        <span class="detail-label">Source:</span>
        <span class="detail-value">
          <span class="engine-badge ${word.engine}">${sourceAbbrev}</span>
          ${sourceName}
        </span>
      </div>
    `;
  }

  // Build harmonization section if available
  const harmonizeHtml = word.harmonize ? buildHarmonizeSection(word.harmonize) : '';

  // Build sanitization section if available
  const sanitizeHtml = word.sanitizeDetails ? buildSanitizeSection(word.sanitizeDetails, word.text) : '';

  wordDetailsBody.innerHTML = `
    <div class="detail-row">
      <span class="detail-label">Word ID:</span>
      <span class="detail-value">${word.wordId}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Text:</span>
      <span class="detail-value text-value">${word.text ? escapeHtml(word.text) : '(empty)'}</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Status:</span>
      <span class="detail-value">${status}</span>
    </div>
    ${sourcesHtml}
    <div class="detail-row">
      <span class="detail-label">Confidence:</span>
      <span class="detail-value">${word.confidence.toFixed(1)}%</span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Bbox:</span>
      <span class="detail-value bbox-value">
        (${word.bbox.x0.toFixed(1)}, ${word.bbox.y0.toFixed(1)}) - (${word.bbox.x1.toFixed(1)}, ${word.bbox.y1.toFixed(1)})
      </span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Size:</span>
      <span class="detail-value">
        ${(word.bbox.x1 - word.bbox.x0).toFixed(1)} × ${(word.bbox.y1 - word.bbox.y0).toFixed(1)} px
      </span>
    </div>
    <div class="detail-row">
      <span class="detail-label">Page:</span>
      <span class="detail-value">${word.page + 1}</span>
    </div>
    ${harmonizeHtml}
    ${sanitizeHtml}
  `;

  wordDetailsModal.classList.remove('hidden');
}

function closeWordDetails() {
  wordDetailsModal.classList.add('hidden');
}

function renderReadingOrder() {
  const visibleEngines = getVisibleEngines();
  const pageWords = extractedWords.filter(
    (w) => w.page === currentPage && shouldShowWord(w, visibleEngines)
  );

  if (pageWords.length === 0) {
    readingOrderContent.innerHTML = '<p class="empty-state">Extract words to see reading order</p>';
    lineCount.textContent = '(0)';
    return;
  }

  // Use cluster-based grouping for visual lines
  // Words are already in correct reading order (by wordId), we just need to group for display
  const lines = groupWordsByCluster(pageWords);
  lineCount.textContent = `(${lines.length})`;

  const showT = showTesseractCheckbox.checked;
  const showD = showDoctrCheckbox.checked;
  const highlightSanitized = highlightSanitizedCheckbox?.checked ?? false;
  const highlightHarmonized = highlightHarmonizedCheckbox?.checked ?? false;

  let html = '';
  for (const line of lines) {
    html += '<div class="reading-line">';
    for (let i = 0; i < line.length; i++) {
      const word = line[i];
      const sources = word.sources ?? [];
      const isMultiSource = sources.length > 1;
      const hasTesseract = sources.includes('tesseract') || word.engine === 'tesseract';
      const hasDoctr = sources.includes('doctr') || word.engine === 'doctr';
      const isPixel = word.engine === 'pixel_detector' || sources.includes('pixel_detector');

      // Determine source style class based on visible toggles
      const sourceVisible = (hasTesseract && showT) || (hasDoctr && showD) || (isPixel && showPixelCheckbox.checked);

      let styleClass: string;
      if (isPixel) {
        styleClass = 'pixel_detector';
      } else if (isMultiSource && showT && showD) {
        styleClass = 'harmonized';
      } else if (hasTesseract && showT) {
        styleClass = 'tesseract';
      } else if (hasDoctr && showD) {
        styleClass = 'doctr';
      } else if (!sourceVisible) {
        // Word visible only due to highlight mode
        styleClass = 'highlight-only';
      } else if (hasTesseract) {
        styleClass = 'tesseract';
      } else if (hasDoctr) {
        styleClass = 'doctr';
      } else {
        styleClass = word.engine;
      }

      const lowConfClass = word.lowConfidence ? ' low-confidence' : '';

      // Add highlight classes
      let highlightClasses = '';
      if (highlightSanitized && word.sanitizeDetails) {
        const status = word.sanitizeDetails.status;
        if (status === 'corrected' || status === 'context') {
          highlightClasses += ' sanitize-corrected';
        } else if (status === 'uncertain') {
          highlightClasses += ' sanitize-uncertain';
        }
      }
      if (highlightHarmonized && isMultiSource) {
        highlightClasses += ' harmonize-matched';
      }

      html += `<span class="reading-word ${styleClass}${lowConfClass}${highlightClasses}" data-word-id="${word.wordId}">${escapeHtml(word.text)}</span>`;
      if (i < line.length - 1) html += ' ';
    }
    html += '</div>';
  }

  readingOrderContent.innerHTML = html;

  // Add event listeners to reading order words
  readingOrderContent.querySelectorAll('.reading-word').forEach((el) => {
    el.addEventListener('mouseenter', () => {
      const wordId = parseInt(el.getAttribute('data-word-id') || '0');
      highlightWordFromList(wordId);
    });
    el.addEventListener('mouseleave', () => unhighlightWord());
    el.addEventListener('click', () => {
      const wordId = parseInt(el.getAttribute('data-word-id') || '0');
      selectWord(wordId);
    });
  });
}

/**
 * Group words into visual lines based on cluster row bands.
 *
 * Uses the cluster data to determine line breaks:
 * - Row bands group clusters that are on the same visual row
 * - Clusters within the same band flow left-to-right on one line
 * - New visual line when moving to a new row band
 *
 * Words are kept in their wordId order (the computed reading order).
 */
function groupWordsByCluster(words: ExtractedWord[]): ExtractedWord[][] {
  if (words.length === 0) return [];

  // If no cluster info, fall back to simple y-based grouping
  if (clusterInfo.length === 0) {
    return groupWordsByYPosition(words);
  }

  // Build a map of wordId -> bandIndex (row band)
  const wordToBand = new Map<number, number>();
  for (const cluster of clusterInfo) {
    for (const word of cluster.words) {
      wordToBand.set(word.wordId, cluster.bandIndex);
    }
  }

  const lines: ExtractedWord[][] = [];
  let currentLine: ExtractedWord[] = [];
  let currentBandIndex: number | undefined = undefined;

  for (const word of words) {
    const bandIndex = wordToBand.get(word.wordId);

    // Start a new line when row band changes
    if (bandIndex !== currentBandIndex && currentLine.length > 0) {
      lines.push(currentLine);
      currentLine = [];
    }

    currentLine.push(word);
    currentBandIndex = bandIndex;
  }

  // Don't forget the last line
  if (currentLine.length > 0) {
    lines.push(currentLine);
  }

  return lines;
}

/**
 * Fallback: group words by y-position when no cluster info available.
 */
function groupWordsByYPosition(words: ExtractedWord[]): ExtractedWord[][] {
  if (words.length === 0) return [];

  // Use y-fuzz from thresholds if available, else default
  const yFuzz = orderingThresholds?.yFuzz ?? 10;

  const lines: ExtractedWord[][] = [];
  let currentLine: ExtractedWord[] = [words[0]];
  let currentYCenter = (words[0].bbox.y0 + words[0].bbox.y1) / 2;

  for (let i = 1; i < words.length; i++) {
    const word = words[i];
    const wordYCenter = (word.bbox.y0 + word.bbox.y1) / 2;

    // Check if this word is on a significantly different row
    if (Math.abs(wordYCenter - currentYCenter) > yFuzz * 2) {
      lines.push(currentLine);
      currentLine = [word];
      currentYCenter = wordYCenter;
    } else {
      currentLine.push(word);
      // Update the row center as a running average
      currentYCenter = (currentYCenter + wordYCenter) / 2;
    }
  }

  if (currentLine.length > 0) {
    lines.push(currentLine);
  }

  return lines;
}

function escapeHtml(text: string): string {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Register service worker for PWA offline support
async function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    try {
      // Use base path for GH Pages compatibility
      const basePath = import.meta.env.BASE_URL || '/';
      const swPath = `${basePath}sw.js`.replace('//', '/');
      const registration = await navigator.serviceWorker.register(swPath, {
        scope: basePath,
      });
      console.log('[PWA] Service Worker registered:', registration.scope);

      // Check for updates
      registration.addEventListener('updatefound', () => {
        const newWorker = registration.installing;
        if (newWorker) {
          newWorker.addEventListener('statechange', () => {
            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
              console.log('[PWA] New version available');
              // Could show update notification here
            }
          });
        }
      });
    } catch (error) {
      console.warn('[PWA] Service Worker registration failed:', error);
    }
  }
}

// Start the app
init();
registerServiceWorker();
