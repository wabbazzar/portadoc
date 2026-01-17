/**
 * Portadoc Web Application
 * PDF visualization with OCR bounding box overlays and bidirectional highlighting
 */

// Global state
const state = {
    pdfDoc: null,
    currentPage: 1,
    totalPages: 0,
    scale: 1.5,
    words: [],
    currentPdfPath: null,
    highlightedWordId: null,
};

// Status colors for bounding boxes
const STATUS_COLORS = {
    word: 'rgba(74, 222, 128, 0.5)',        // green
    low_conf: 'rgba(250, 204, 21, 0.5)',    // yellow
    secondary_only: 'rgba(251, 146, 60, 0.5)', // orange
    pixel: 'rgba(248, 113, 113, 0.5)',      // red
};

const STATUS_BORDERS = {
    word: '#4ade80',
    low_conf: '#facc15',
    secondary_only: '#fb923c',
    pixel: '#f87171',
};

// DOM elements
let pdfCanvas, bboxCanvas, pdfCtx, bboxCtx;
let pdfSelect, extractBtn, wordsTable, wordsTbody;
let pageInfo, prevPageBtn, nextPageBtn;
let configPanel, toggleConfigBtn;
let statusFilter;

// Initialize PDF.js
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

document.addEventListener('DOMContentLoaded', init);

async function init() {
    // Get DOM elements
    pdfCanvas = document.getElementById('pdf-canvas');
    bboxCanvas = document.getElementById('bbox-canvas');
    pdfCtx = pdfCanvas.getContext('2d');
    bboxCtx = bboxCanvas.getContext('2d');

    pdfSelect = document.getElementById('pdf-select');
    extractBtn = document.getElementById('extract-btn');
    wordsTable = document.getElementById('words-table');
    wordsTbody = document.getElementById('words-tbody');

    pageInfo = document.getElementById('page-info');
    prevPageBtn = document.getElementById('prev-page');
    nextPageBtn = document.getElementById('next-page');

    configPanel = document.getElementById('config-panel');
    toggleConfigBtn = document.getElementById('toggle-config-btn');
    statusFilter = document.getElementById('status-filter');

    // Event listeners
    pdfSelect.addEventListener('change', onPdfSelect);
    extractBtn.addEventListener('click', onExtract);
    prevPageBtn.addEventListener('click', () => changePage(-1));
    nextPageBtn.addEventListener('click', () => changePage(1));
    toggleConfigBtn.addEventListener('click', toggleConfig);
    statusFilter.addEventListener('change', () => {
        renderWordList();
        renderBboxes();
    });
    document.getElementById('apply-extract-btn').addEventListener('click', onExtract);
    document.getElementById('close-details').addEventListener('click', closeDetails);

    // Bbox canvas mouse events for hover
    bboxCanvas.style.pointerEvents = 'auto';
    bboxCanvas.addEventListener('mousemove', onCanvasMouseMove);
    bboxCanvas.addEventListener('click', onCanvasClick);
    bboxCanvas.addEventListener('mouseleave', () => highlightWord(null));

    // Load PDF list
    await loadPdfList();
}

async function loadPdfList() {
    try {
        const response = await fetch('/api/pdfs');
        const data = await response.json();

        pdfSelect.innerHTML = '<option value="">Select PDF...</option>';
        for (const pdf of data.pdfs) {
            const option = document.createElement('option');
            option.value = pdf.path;
            option.textContent = pdf.name + (pdf.has_extraction ? ' [extracted]' : '');
            pdfSelect.appendChild(option);
        }
    } catch (error) {
        console.error('Failed to load PDF list:', error);
    }
}

async function onPdfSelect() {
    const pdfPath = pdfSelect.value;
    if (!pdfPath) return;

    state.currentPdfPath = pdfPath;
    extractBtn.disabled = false;

    // Load PDF
    await loadPdf(pdfPath);

    // Load words if extraction exists
    await loadWords(pdfPath);
}

async function loadPdf(pdfPath) {
    try {
        const loadingTask = pdfjsLib.getDocument(`/api/pdf/${encodeURIComponent(pdfPath)}`);
        state.pdfDoc = await loadingTask.promise;
        state.totalPages = state.pdfDoc.numPages;
        state.currentPage = 1;

        updatePageControls();
        await renderPage(state.currentPage);
    } catch (error) {
        console.error('Failed to load PDF:', error);
    }
}

async function loadWords(pdfPath) {
    try {
        const response = await fetch(`/api/words/${encodeURIComponent(pdfPath)}`);
        const data = await response.json();

        state.words = data.words || [];
        renderWordList();
        renderBboxes();

        // Update stats
        const statsDiv = document.getElementById('stats');
        if (data.extracted) {
            const statusCounts = data.status_counts || {};
            const sourceCounts = data.source_counts || {};
            statsDiv.innerHTML = `
                <strong>Total:</strong> ${data.total_words} words |
                <span class="status-word">word: ${statusCounts.word || 0}</span> |
                <span class="status-low_conf">low_conf: ${statusCounts.low_conf || 0}</span> |
                <span class="status-secondary_only">secondary: ${statusCounts.secondary_only || 0}</span>
            `;
        } else {
            statsDiv.innerHTML = '<em>No extraction found. Click Extract to run.</em>';
        }

        document.getElementById('word-count').textContent = `(${state.words.length})`;
    } catch (error) {
        console.error('Failed to load words:', error);
    }
}

async function renderPage(pageNum) {
    if (!state.pdfDoc) return;

    const page = await state.pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: state.scale });

    // Set canvas dimensions
    pdfCanvas.width = viewport.width;
    pdfCanvas.height = viewport.height;
    bboxCanvas.width = viewport.width;
    bboxCanvas.height = viewport.height;

    // Position bbox canvas over pdf canvas
    const container = document.getElementById('pdf-container');
    bboxCanvas.style.left = pdfCanvas.offsetLeft + 'px';
    bboxCanvas.style.top = pdfCanvas.offsetTop + 'px';

    // Render PDF page
    const renderContext = {
        canvasContext: pdfCtx,
        viewport: viewport,
    };

    await page.render(renderContext).promise;
    renderBboxes();
}

function renderBboxes() {
    if (!bboxCtx) return;

    bboxCtx.clearRect(0, 0, bboxCanvas.width, bboxCanvas.height);

    // Get page dimensions from PDF (assuming 72 DPI base)
    const scale = state.scale;
    const filterValue = statusFilter ? statusFilter.value : '';
    const pageWords = state.words.filter(w =>
        w.page === state.currentPage - 1 &&
        (!filterValue || w.status === filterValue)
    );

    for (const word of pageWords) {
        const status = word.status || 'word';
        const fillColor = STATUS_COLORS[status] || STATUS_COLORS.word;
        const borderColor = STATUS_BORDERS[status] || STATUS_BORDERS.word;

        // Scale coordinates (PDF coords are in points, canvas is scaled)
        const x = word.x0 * scale;
        const y = word.y0 * scale;
        const w = (word.x1 - word.x0) * scale;
        const h = (word.y1 - word.y0) * scale;

        // Highlight effect
        const isHighlighted = state.highlightedWordId === word.word_id;

        bboxCtx.fillStyle = isHighlighted ? borderColor : fillColor;
        bboxCtx.fillRect(x, y, w, h);

        bboxCtx.strokeStyle = borderColor;
        bboxCtx.lineWidth = isHighlighted ? 3 : 1;
        bboxCtx.strokeRect(x, y, w, h);
    }
}

function renderWordList() {
    const filterValue = statusFilter.value;
    const pageWords = state.words.filter(w =>
        w.page === state.currentPage - 1 &&
        (!filterValue || w.status === filterValue)
    );

    wordsTbody.innerHTML = '';

    for (const word of pageWords) {
        const tr = document.createElement('tr');
        tr.dataset.wordId = word.word_id;
        tr.classList.add(`status-${word.status}`);

        tr.innerHTML = `
            <td>${word.word_id}</td>
            <td>${escapeHtml(word.text)}</td>
            <td>${word.source}</td>
            <td class="status-${word.status}">${word.status}</td>
            <td>${word.confidence.toFixed(1)}</td>
        `;

        tr.addEventListener('mouseenter', () => highlightWord(word.word_id));
        tr.addEventListener('mouseleave', () => highlightWord(null));
        tr.addEventListener('click', () => showWordDetails(word));

        wordsTbody.appendChild(tr);
    }

    document.getElementById('word-count').textContent = `(${pageWords.length}/${state.words.length})`;
}

function highlightWord(wordId) {
    state.highlightedWordId = wordId;
    renderBboxes();

    // Highlight table row
    document.querySelectorAll('#words-tbody tr').forEach(tr => {
        tr.classList.toggle('highlighted', tr.dataset.wordId == wordId);
    });
}

function onCanvasMouseMove(event) {
    const rect = bboxCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / state.scale;
    const y = (event.clientY - rect.top) / state.scale;

    const pageWords = state.words.filter(w => w.page === state.currentPage - 1);

    for (const word of pageWords) {
        if (x >= word.x0 && x <= word.x1 && y >= word.y0 && y <= word.y1) {
            highlightWord(word.word_id);
            bboxCanvas.style.cursor = 'pointer';
            return;
        }
    }

    highlightWord(null);
    bboxCanvas.style.cursor = 'default';
}

function onCanvasClick(event) {
    const rect = bboxCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) / state.scale;
    const y = (event.clientY - rect.top) / state.scale;

    const pageWords = state.words.filter(w => w.page === state.currentPage - 1);

    for (const word of pageWords) {
        if (x >= word.x0 && x <= word.x1 && y >= word.y0 && y <= word.y1) {
            showWordDetails(word);
            return;
        }
    }
}

// Source code legend
const SOURCE_NAMES = {
    'T': 'Tesseract',
    'E': 'EasyOCR',
    'D': 'docTR',
    'P': 'PaddleOCR',
    'PX': 'Pixel Detection',
    'X': 'Pixel Detection',  // Legacy
};

function decodeSource(source) {
    if (!source) return '-';
    // Handle "PX" as a unit first
    if (source === 'PX' || source === 'X') {
        return SOURCE_NAMES[source];
    }
    // Decode each letter for combined sources like "TE", "TEP"
    const parts = source.split('').map(c => SOURCE_NAMES[c] || c);
    return parts.join(' + ');
}

function showWordDetails(word) {
    const detailsBody = document.getElementById('details-body');
    detailsBody.innerHTML = `
        <p><strong>Word ID:</strong> ${word.word_id}</p>
        <p><strong>Text:</strong> ${escapeHtml(word.text)}</p>
        <p><strong>Status:</strong> <span class="status-${word.status}">${word.status}</span></p>
        <p><strong>Source:</strong> ${decodeSource(word.source)} <span style="opacity: 0.6">(${word.source})</span></p>
        <p><strong>Confidence:</strong> ${word.confidence.toFixed(1)}%</p>
        <p><strong>Bbox:</strong> (${word.x0.toFixed(1)}, ${word.y0.toFixed(1)}) - (${word.x1.toFixed(1)}, ${word.y1.toFixed(1)})</p>
        <hr style="margin: 1rem 0; border-color: var(--bg-panel);">
        <p><strong>Tesseract:</strong> ${escapeHtml(word.tess_text || '-')}</p>
        <p><strong>EasyOCR:</strong> ${escapeHtml(word.easy_text || '-')}</p>
        <p><strong>docTR:</strong> ${escapeHtml(word.doctr_text || '-')}</p>
        <p><strong>PaddleOCR:</strong> ${escapeHtml(word.paddle_text || '-')}</p>
    `;

    document.getElementById('word-details').classList.remove('hidden');
}

function closeDetails() {
    document.getElementById('word-details').classList.add('hidden');
}

async function onExtract() {
    if (!state.currentPdfPath) return;

    extractBtn.disabled = true;
    extractBtn.textContent = 'Extracting...';
    document.body.classList.add('loading');

    const request = {
        pdf_path: state.currentPdfPath,
        use_tesseract: true,
        use_easyocr: document.getElementById('cfg-easyocr').checked,
        use_paddleocr: document.getElementById('cfg-paddleocr').checked,
        use_doctr: document.getElementById('cfg-doctr').checked,
        preprocess: document.getElementById('cfg-preprocess').value,
        psm: parseInt(document.getElementById('cfg-psm').value),
        oem: parseInt(document.getElementById('cfg-oem').value),
    };

    try {
        const response = await fetch('/api/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });

        const data = await response.json();

        if (data.success) {
            state.words = data.words || [];
            renderWordList();
            renderBboxes();
            alert(`Extracted ${data.total_words} words`);
        } else {
            alert('Extraction failed: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Extraction failed:', error);
        alert('Extraction failed: ' + error.message);
    } finally {
        extractBtn.disabled = false;
        extractBtn.textContent = 'Extract';
        document.body.classList.remove('loading');
    }
}

function changePage(delta) {
    const newPage = state.currentPage + delta;
    if (newPage >= 1 && newPage <= state.totalPages) {
        state.currentPage = newPage;
        updatePageControls();
        renderPage(state.currentPage);
        renderWordList();
    }
}

function updatePageControls() {
    pageInfo.textContent = `Page ${state.currentPage} / ${state.totalPages}`;
    prevPageBtn.disabled = state.currentPage <= 1;
    nextPageBtn.disabled = state.currentPage >= state.totalPages;
}

function toggleConfig() {
    configPanel.classList.toggle('hidden');
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
