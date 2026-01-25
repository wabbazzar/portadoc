/**
 * Portadoc Browser - Main Application
 * Client-side PDF word extraction with OCR
 */

import { Word, ExtractedWord } from './models';
import { loadPdf, renderPage, getPdfPageCount } from './pdf-loader';
import { extractWithTesseract } from './ocr/tesseract';
// import { extractWithDoctr } from './ocr/doctr';
import { orderWordsByReading } from './geometric-clustering';
// import { harmonizeWords } from './harmonize';

// State
let currentFile: File | null = null;
let pdfDocument: Awaited<ReturnType<typeof loadPdf>> | null = null;
let currentPage = 0;
let totalPages = 1;
let extractedWords: ExtractedWord[] = [];
let selectedWordId: number | null = null;

// DOM Elements
const dropZone = document.getElementById('drop-zone')!;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const controlsSection = document.getElementById('controls-section')!;
const progressSection = document.getElementById('progress-section')!;
const viewerSection = document.getElementById('viewer-section')!;
const pdfCanvas = document.getElementById('pdf-canvas') as HTMLCanvasElement;
const bboxOverlay = document.getElementById('bbox-overlay')!;
const wordsList = document.getElementById('words-list')!;
const wordCount = document.getElementById('word-count')!;
const pageInfo = document.getElementById('page-info')!;
const prevPageBtn = document.getElementById('prev-page') as HTMLButtonElement;
const nextPageBtn = document.getElementById('next-page') as HTMLButtonElement;
const extractBtn = document.getElementById('extract-btn') as HTMLButtonElement;
const useTesseractCheckbox = document.getElementById('use-tesseract') as HTMLInputElement;
const useDoctrCheckbox = document.getElementById('use-doctr') as HTMLInputElement;
const exportCsvBtn = document.getElementById('export-csv')!;
const exportJsonBtn = document.getElementById('export-json')!;
const progressFill = document.getElementById('progress-fill')!;
const progressText = document.getElementById('progress-text')!;

// Initialize
function init() {
  setupEventListeners();
}

function setupEventListeners() {
  // File upload
  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  dropZone.addEventListener('drop', handleFileDrop);
  fileInput.addEventListener('change', handleFileSelect);

  // Page navigation
  prevPageBtn.addEventListener('click', () => navigatePage(-1));
  nextPageBtn.addEventListener('click', () => navigatePage(1));

  // Extraction
  extractBtn.addEventListener('click', handleExtract);

  // Export
  exportCsvBtn.addEventListener('click', exportCsv);
  exportJsonBtn.addEventListener('click', exportJson);
}

function handleFileDrop(e: DragEvent) {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const files = e.dataTransfer?.files;
  if (files && files.length > 0) {
    processFile(files[0]);
  }
}

function handleFileSelect(e: Event) {
  const input = e.target as HTMLInputElement;
  if (input.files && input.files.length > 0) {
    processFile(input.files[0]);
  }
}

async function processFile(file: File) {
  currentFile = file;
  currentPage = 0;
  extractedWords = [];

  showProgress('Loading document...');

  try {
    if (file.type === 'application/pdf') {
      pdfDocument = await loadPdf(file);
      totalPages = getPdfPageCount(pdfDocument);
    } else {
      // Image file - treat as single page
      pdfDocument = null;
      totalPages = 1;
    }

    updatePageControls();
    await renderCurrentPage();

    hideProgress();
    controlsSection.classList.remove('hidden');
    viewerSection.classList.remove('hidden');
  } catch (error) {
    console.error('Error loading file:', error);
    hideProgress();
    alert('Error loading file: ' + (error as Error).message);
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

  // Re-render bboxes for current page
  renderBboxes();
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
  pageInfo.textContent = `Page ${currentPage + 1} of ${totalPages}`;
  prevPageBtn.disabled = currentPage === 0;
  nextPageBtn.disabled = currentPage >= totalPages - 1;
}

async function handleExtract() {
  if (!currentFile) return;

  const useTesseract = useTesseractCheckbox.checked;
  const useDoctr = useDoctrCheckbox.checked;

  if (!useTesseract && !useDoctr) {
    alert('Please select at least one OCR engine');
    return;
  }

  extractBtn.disabled = true;
  showProgress('Initializing OCR...');
  extractedWords = [];

  try {
    // Get image data from canvas
    const imageData = pdfCanvas.toDataURL('image/png');

    // Run OCR engines
    const allWords: Word[] = [];

    if (useTesseract) {
      showProgress('Running Tesseract.js...');
      const tesseractWords = await extractWithTesseract(
        imageData,
        currentPage,
        (progress) => {
          updateProgress(progress * (useDoctr ? 50 : 100), `Tesseract: ${Math.round(progress * 100)}%`);
        }
      );
      allWords.push(...tesseractWords);
    }

    if (useDoctr) {
      showProgress('docTR not yet implemented...');
      // TODO: Implement docTR extraction
      // const doctrWords = await extractWithDoctr(imageData, currentPage, (progress) => {
      //   updateProgress(50 + progress * 50, `docTR: ${Math.round(progress * 100)}%`);
      // });
      // allWords.push(...doctrWords);
    }

    // Order words by reading order
    showProgress('Ordering words...');
    const orderedWords = orderWordsByReading(allWords);

    // Convert to ExtractedWord
    extractedWords = orderedWords.map((w, i) => ({
      ...w,
      wordId: i,
    }));

    renderBboxes();
    renderWordsList();
    hideProgress();
  } catch (error) {
    console.error('Extraction error:', error);
    alert('Error during extraction: ' + (error as Error).message);
    hideProgress();
  } finally {
    extractBtn.disabled = false;
  }
}

function renderBboxes() {
  bboxOverlay.innerHTML = '';
  bboxOverlay.style.width = pdfCanvas.width + 'px';
  bboxOverlay.style.height = pdfCanvas.height + 'px';

  const pageWords = extractedWords.filter((w) => w.page === currentPage);

  for (const word of pageWords) {
    const div = document.createElement('div');
    div.className = 'bbox';
    if (word.engine === 'tesseract') div.classList.add('tesseract');
    if (word.engine === 'doctr') div.classList.add('doctr');
    if (selectedWordId === word.wordId) div.classList.add('selected');

    div.style.left = word.bbox.x0 + 'px';
    div.style.top = word.bbox.y0 + 'px';
    div.style.width = (word.bbox.x1 - word.bbox.x0) + 'px';
    div.style.height = (word.bbox.y1 - word.bbox.y0) + 'px';

    div.dataset.wordId = String(word.wordId);
    div.addEventListener('click', () => selectWord(word.wordId));

    bboxOverlay.appendChild(div);
  }
}

function renderWordsList() {
  wordsList.innerHTML = '';
  const pageWords = extractedWords.filter((w) => w.page === currentPage);
  wordCount.textContent = `${pageWords.length} words`;

  for (const word of pageWords) {
    const div = document.createElement('div');
    div.className = 'word-item';
    if (selectedWordId === word.wordId) div.classList.add('selected');

    const textSpan = document.createElement('span');
    textSpan.className = 'word-text';
    textSpan.textContent = word.text;

    const engineSpan = document.createElement('span');
    engineSpan.className = `word-engine ${word.engine}`;
    engineSpan.textContent = word.engine.charAt(0).toUpperCase();

    div.appendChild(textSpan);
    div.appendChild(engineSpan);
    div.addEventListener('click', () => selectWord(word.wordId));

    wordsList.appendChild(div);
  }
}

function selectWord(wordId: number) {
  selectedWordId = selectedWordId === wordId ? null : wordId;
  renderBboxes();
  renderWordsList();

  // Scroll word into view in list
  if (selectedWordId !== null) {
    const wordItem = wordsList.querySelector(`.word-item:nth-child(${selectedWordId + 1})`);
    wordItem?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function showProgress(text: string) {
  progressSection.classList.remove('hidden');
  progressText.textContent = text;
  progressFill.style.width = '0%';
}

function updateProgress(percent: number, text: string) {
  progressFill.style.width = percent + '%';
  progressText.textContent = text;
}

function hideProgress() {
  progressSection.classList.add('hidden');
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

// Start the app
init();
