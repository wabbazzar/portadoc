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
    readingOrderLines: [],
    currentPdfPath: null,
    highlightedWordId: null,
    lastExtractionInfo: null, // { pdfPath, csvPath, wordCount }
    pageRotations: {}, // Page rotation angles from extraction (0, 90, 180, 270)
    uiDescriptions: {}, // Tooltip descriptions from config
    entityStats: { total: 0, names: 0, dates: 0, codes: 0 }, // Redaction stats
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
let pdfSelect, pdfUpload, extractBtn, wordsTable, wordsTbody;
let pageInfo, prevPageBtn, nextPageBtn;
let configPanel, toggleConfigBtn;
let statusFilter, sourceFilter, wordTooltip;
let readingOrderContent, lineCount;

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
    pdfUpload = document.getElementById('pdf-upload');
    extractBtn = document.getElementById('extract-btn');
    wordsTable = document.getElementById('words-table');
    wordsTbody = document.getElementById('words-tbody');

    pageInfo = document.getElementById('page-info');
    prevPageBtn = document.getElementById('prev-page');
    nextPageBtn = document.getElementById('next-page');

    configPanel = document.getElementById('config-panel');
    toggleConfigBtn = document.getElementById('toggle-config-btn');
    statusFilter = document.getElementById('status-filter');
    sourceFilter = document.getElementById('source-filter');
    wordTooltip = document.getElementById('word-tooltip');

    // Event listeners
    pdfSelect.addEventListener('change', onPdfSelect);
    pdfUpload.addEventListener('change', onPdfUpload);
    extractBtn.addEventListener('click', onExtract);
    prevPageBtn.addEventListener('click', () => changePage(-1));
    nextPageBtn.addEventListener('click', () => changePage(1));
    toggleConfigBtn.addEventListener('click', toggleConfig);
    statusFilter.addEventListener('change', () => {
        renderWordList();
        renderBboxes();
    });
    sourceFilter.addEventListener('change', () => {
        renderWordList();
        renderBboxes();
    });
    document.getElementById('apply-extract-btn').addEventListener('click', onExtract);
    document.getElementById('close-details').addEventListener('click', closeDetails);

    // File info modal
    document.getElementById('file-info-btn').addEventListener('click', showFileInfoModal);
    document.getElementById('close-file-info').addEventListener('click', () => {
        document.getElementById('file-info-modal').classList.add('hidden');
    });
    document.getElementById('file-info-modal').addEventListener('click', (e) => {
        if (e.target.id === 'file-info-modal') {
            document.getElementById('file-info-modal').classList.add('hidden');
        }
    });

    // Redaction button
    document.getElementById('apply-redactions-btn')?.addEventListener('click', applyRedactions);

    // Bbox canvas mouse events for hover
    bboxCanvas.style.pointerEvents = 'auto';
    bboxCanvas.addEventListener('mousemove', onCanvasMouseMove);
    bboxCanvas.addEventListener('click', onCanvasClick);
    bboxCanvas.addEventListener('mouseleave', () => {
        highlightWord(null);
        hideTooltip();
    });

    // Initialize panel resizers
    initPanelResizer();
    initHorizontalResizer();

    // Get reading order elements
    readingOrderContent = document.getElementById('reading-order-content');
    lineCount = document.getElementById('line-count');

    // Load PDF list and config
    await loadPdfList();
    await loadConfig();
}

async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();

        // Populate harmonization fields
        if (config.harmonize) {
            setInputValue('cfg-iou-threshold', config.harmonize.iou_threshold);
            setInputValue('cfg-text-match-bonus', config.harmonize.text_match_bonus);
            setInputValue('cfg-center-distance-max', config.harmonize.center_distance_max);
            setInputValue('cfg-word-min-conf', config.harmonize.word_min_conf);
            setInputValue('cfg-low-conf-min', config.harmonize.low_conf_min_conf);
        }

        // Populate geometric clustering fields
        if (config.geometric_clustering) {
            if (config.geometric_clustering.y_fuzz) {
                setInputValue('cfg-yfuzz-default', config.geometric_clustering.y_fuzz.default);
                setInputValue('cfg-yfuzz-multiplier', config.geometric_clustering.y_fuzz.multiplier);
                setInputValue('cfg-yfuzz-max-height-ratio', config.geometric_clustering.y_fuzz.max_height_ratio);
            }
            if (config.geometric_clustering.connection) {
                setInputValue('cfg-x-overlap-min', config.geometric_clustering.connection.x_overlap_min);
                setInputValue('cfg-y-overlap-min', config.geometric_clustering.connection.y_overlap_min);
            }
        }

        // Populate OCR settings
        if (config.ocr && config.ocr.tesseract) {
            setInputValue('cfg-psm', config.ocr.tesseract.psm);
            setInputValue('cfg-oem', config.ocr.tesseract.oem);
        }

        // Store UI descriptions and apply tooltips
        if (config.ui_descriptions) {
            state.uiDescriptions = config.ui_descriptions;
            applyConfigTooltips();
        }

        console.log('Config loaded:', config);
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

function applyConfigTooltips() {
    // Find all elements with data-tooltip-key attribute
    document.querySelectorAll('[data-tooltip-key]').forEach(el => {
        const key = el.dataset.tooltipKey;
        const description = getDescriptionByKey(key);

        if (description) {
            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.className = 'config-tooltip';
            tooltip.textContent = description;

            // For grid items (checkboxes), position relative to parent
            if (el.classList.contains('config-engines-grid')) {
                // This is the grid container, skip it
                return;
            }

            // Insert tooltip into the element
            el.style.position = 'relative';
            el.appendChild(tooltip);
        }
    });
}

function getDescriptionByKey(key) {
    // Key format: "category.field" e.g., "ocr_engines.primary_engine"
    const parts = key.split('.');
    if (parts.length !== 2) return null;

    const [category, field] = parts;
    if (state.uiDescriptions[category] && state.uiDescriptions[category][field]) {
        return state.uiDescriptions[category][field];
    }
    return null;
}

function setInputValue(id, value) {
    const el = document.getElementById(id);
    if (el && value !== undefined && value !== null) {
        el.value = value;
    }
}

function getFloatValue(id) {
    const el = document.getElementById(id);
    if (el && el.value) {
        const val = parseFloat(el.value);
        return isNaN(val) ? null : val;
    }
    return null;
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

    // Clear file input when selecting from dropdown
    pdfUpload.value = '';

    state.currentPdfPath = pdfPath;
    extractBtn.disabled = false;

    // Load PDF
    await loadPdf(pdfPath);

    // Load words if extraction exists
    await loadWords(pdfPath);
}

async function onPdfUpload() {
    const file = pdfUpload.files[0];
    if (!file) return;

    // Clear dropdown selection
    pdfSelect.value = '';

    // Show upload progress
    extractBtn.disabled = true;
    extractBtn.textContent = 'Uploading...';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            state.currentPdfPath = data.path;
            extractBtn.disabled = false;
            extractBtn.textContent = 'Extract';

            // Load the uploaded PDF
            await loadPdf(data.path);

            // Check for existing extraction
            await loadWords(data.path);

            // Refresh PDF list to show the uploaded file
            await loadPdfList();
        } else {
            alert('Upload failed: ' + (data.detail || 'Unknown error'));
            extractBtn.textContent = 'Extract';
        }
    } catch (error) {
        console.error('Upload failed:', error);
        alert('Upload failed: ' + error.message);
        extractBtn.textContent = 'Extract';
    }
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
        state.pageRotations = data.page_rotations || {};

        // Re-render page if rotation is non-zero (PDF needs to be rotated)
        const pageIndex = state.currentPage - 1;
        const rotation = state.pageRotations[pageIndex] || 0;
        if (rotation !== 0) {
            await renderPage(state.currentPage);
        } else {
            renderBboxes();
        }
        renderWordList();

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
            // Also load reading order
            await loadReadingOrder(pdfPath);
        } else {
            statsDiv.innerHTML = '<em>No extraction found. Click Extract to run.</em>';
            renderReadingOrderEmpty();
        }

        document.getElementById('word-count').textContent = `(${state.words.length})`;
    } catch (error) {
        console.error('Failed to load words:', error);
    }
}

async function loadReadingOrder(pdfPath) {
    try {
        const page = state.currentPage - 1; // API uses 0-indexed pages
        const response = await fetch(`/api/reading-order/${encodeURIComponent(pdfPath)}?page=${page}`);
        const data = await response.json();

        state.readingOrderLines = data.lines || [];
        renderReadingOrder();

        if (lineCount) {
            lineCount.textContent = `(${data.total_lines || 0} lines)`;
        }
    } catch (error) {
        console.error('Failed to load reading order:', error);
        renderReadingOrderEmpty();
    }
}

function renderReadingOrderEmpty() {
    if (readingOrderContent) {
        readingOrderContent.innerHTML = '<p class="empty-state">Load a PDF with extraction to see reading order.</p>';
    }
    if (lineCount) {
        lineCount.textContent = '(0 lines)';
    }
    state.readingOrderLines = [];
}

function renderReadingOrder() {
    if (!readingOrderContent) return;

    if (state.readingOrderLines.length === 0) {
        renderReadingOrderEmpty();
        return;
    }

    let html = '';
    for (const line of state.readingOrderLines) {
        // Build line with clickable words
        let lineHtml = `<div class="reading-line" data-line-id="${line.line_id}" data-word-ids="${line.word_ids.join(',')}">`;

        for (let i = 0; i < line.words.length; i++) {
            const word = line.words[i];
            lineHtml += `<span class="reading-word status-${word.status}" data-word-id="${word.word_id}">${escapeHtml(word.text)}</span>`;
            if (i < line.words.length - 1) {
                lineHtml += ' ';
            }
        }

        lineHtml += '</div>';
        html += lineHtml;
    }

    readingOrderContent.innerHTML = html;

    // Add event listeners to words
    readingOrderContent.querySelectorAll('.reading-word').forEach(el => {
        el.addEventListener('mouseenter', () => {
            const wordId = parseInt(el.dataset.wordId);
            highlightWord(wordId);
        });
        el.addEventListener('mouseleave', () => {
            highlightWord(null);
        });
        el.addEventListener('click', () => {
            const wordId = parseInt(el.dataset.wordId);
            const word = state.words.find(w => w.word_id === wordId);
            if (word) {
                showWordDetails(word);
            }
        });
    });
}

async function renderPage(pageNum) {
    if (!state.pdfDoc) return;

    const page = await state.pdfDoc.getPage(pageNum);

    // Get rotation for this page (API uses 0-indexed pages)
    const pageIndex = pageNum - 1;
    const rotation = state.pageRotations[pageIndex] || 0;

    // Apply rotation to viewport so PDF display matches rotated bboxes
    const viewport = page.getViewport({ scale: state.scale, rotation: rotation });

    // Set canvas dimensions
    pdfCanvas.width = viewport.width;
    pdfCanvas.height = viewport.height;
    bboxCanvas.width = viewport.width;
    bboxCanvas.height = viewport.height;

    // bbox canvas is now positioned via CSS (absolute, top: 0, left: 0)

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
    const statusFilterValue = statusFilter ? statusFilter.value : '';
    const sourceFilterValue = sourceFilter ? sourceFilter.value : '';
    const pageWords = state.words.filter(w =>
        w.page === state.currentPage - 1 &&
        (!statusFilterValue || w.status === statusFilterValue) &&
        (!sourceFilterValue || (w.source && w.source.includes(sourceFilterValue)))
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
    const statusFilterValue = statusFilter.value;
    const sourceFilterValue = sourceFilter.value;
    const pageWords = state.words.filter(w =>
        w.page === state.currentPage - 1 &&
        (!statusFilterValue || w.status === statusFilterValue) &&
        (!sourceFilterValue || (w.source && w.source.includes(sourceFilterValue)))
    );

    wordsTbody.innerHTML = '';

    for (const word of pageWords) {
        const tr = document.createElement('tr');
        tr.dataset.wordId = word.word_id;
        tr.classList.add(`status-${word.status}`);

        const entityBadge = word.entity
            ? `<span class="entity-badge ${word.entity.toLowerCase()}">${word.entity}</span>`
            : '-';

        tr.innerHTML = `
            <td>${word.word_id}</td>
            <td>${escapeHtml(word.text)}</td>
            <td>${word.source}</td>
            <td class="status-${word.status}">${word.status}</td>
            <td>${word.confidence.toFixed(1)}</td>
            <td>${entityBadge}</td>
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

    // Highlight reading order word
    document.querySelectorAll('.reading-word').forEach(el => {
        el.classList.toggle('highlighted', parseInt(el.dataset.wordId) === wordId);
    });

    // Highlight entire line if word is part of it
    document.querySelectorAll('.reading-line').forEach(el => {
        const lineWordIds = el.dataset.wordIds.split(',').map(id => parseInt(id));
        el.classList.toggle('highlighted', lineWordIds.includes(wordId));
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
            showTooltip(word.text, event.clientX - rect.left, event.clientY - rect.top);
            bboxCanvas.style.cursor = 'pointer';
            return;
        }
    }

    highlightWord(null);
    hideTooltip();
    bboxCanvas.style.cursor = 'default';
}

function showTooltip(text, x, y) {
    wordTooltip.textContent = text;
    wordTooltip.style.left = (x + 10) + 'px';
    wordTooltip.style.top = (y - 30) + 'px';
    wordTooltip.classList.remove('hidden');
}

function hideTooltip() {
    wordTooltip.classList.add('hidden');
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
    'S': 'Surya',
    'K': 'Kraken',
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
        <p><strong>Surya:</strong> ${escapeHtml(word.surya_text || '-')}</p>
        <p><strong>Kraken:</strong> ${escapeHtml(word.kraken_text || '-')}</p>
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

    const primaryEngine = document.getElementById('cfg-primary')?.value || null;
    const request = {
        pdf_path: state.currentPdfPath,
        use_tesseract: document.getElementById('cfg-tesseract')?.checked ?? true,
        use_easyocr: document.getElementById('cfg-easyocr')?.checked ?? true,
        use_paddleocr: document.getElementById('cfg-paddleocr')?.checked ?? true,
        use_doctr: document.getElementById('cfg-doctr')?.checked ?? true,
        use_surya: document.getElementById('cfg-surya')?.checked ?? true,
        use_kraken: document.getElementById('cfg-kraken')?.checked ?? false,
        preprocess: document.getElementById('cfg-preprocess')?.value ?? 'none',
        psm: parseInt(document.getElementById('cfg-psm')?.value ?? '6'),
        oem: parseInt(document.getElementById('cfg-oem')?.value ?? '3'),
        primary_engine: primaryEngine,
        // Config overrides
        harmonize: {
            iou_threshold: getFloatValue('cfg-iou-threshold'),
            text_match_bonus: getFloatValue('cfg-text-match-bonus'),
            center_distance_max: getFloatValue('cfg-center-distance-max'),
            word_min_conf: getFloatValue('cfg-word-min-conf'),
            low_conf_min_conf: getFloatValue('cfg-low-conf-min'),
        },
        geometric_clustering: {
            y_fuzz_default: getFloatValue('cfg-yfuzz-default'),
            y_fuzz_multiplier: getFloatValue('cfg-yfuzz-multiplier'),
            y_fuzz_max_height_ratio: getFloatValue('cfg-yfuzz-max-height-ratio'),
            x_overlap_min: getFloatValue('cfg-x-overlap-min'),
            y_overlap_min: getFloatValue('cfg-y-overlap-min'),
        },
    };
    console.log('Extraction request:', request);

    try {
        const response = await fetch('/api/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(request),
        });

        const data = await response.json();

        if (data.success) {
            state.words = data.words || [];
            // Store extraction info for the info modal
            state.lastExtractionInfo = {
                pdfPath: data.pdf_path || state.currentPdfPath,
                csvPath: data.csv_path || 'Unknown',
                wordCount: data.total_words || state.words.length,
            };
            // Show the info button
            document.getElementById('file-info-btn').classList.remove('hidden');

            // Run entity detection if enabled
            const useEntityDetect = document.getElementById('use-entity-detect')?.checked;
            if (useEntityDetect) {
                await runEntityDetection();
            } else {
                hideRedactionSection();
            }

            renderWordList();
            renderBboxes();
            // Reload reading order after extraction
            await loadReadingOrder(state.currentPdfPath);
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

async function changePage(delta) {
    const newPage = state.currentPage + delta;
    if (newPage >= 1 && newPage <= state.totalPages) {
        state.currentPage = newPage;
        updatePageControls();
        await renderPage(state.currentPage);
        renderWordList();
        // Reload reading order for new page
        if (state.currentPdfPath && state.words.length > 0) {
            await loadReadingOrder(state.currentPdfPath);
        }
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

// Panel resizing functionality
function initPanelResizer() {
    const resizer = document.getElementById('panel-resizer');
    const wordList = document.getElementById('word-list');

    if (!resizer || !wordList) return;

    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = wordList.offsetWidth;

        resizer.classList.add('dragging');
        document.body.classList.add('resizing');

        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        // Calculate new width (dragging left increases width, right decreases)
        const deltaX = startX - e.clientX;
        const newWidth = Math.min(Math.max(startWidth + deltaX, 280), 800);

        wordList.style.width = newWidth + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove('dragging');
            document.body.classList.remove('resizing');
        }
    });
}

// Horizontal resizer for Words Panel / Reading Order split
function initHorizontalResizer() {
    const resizer = document.getElementById('horizontal-resizer');
    const wordsPanel = document.getElementById('words-panel');
    const readingPanel = document.getElementById('reading-order-panel');
    const wordList = document.getElementById('word-list');

    if (!resizer || !wordsPanel || !readingPanel || !wordList) return;

    let isResizing = false;
    let startY = 0;
    let startWordsHeight = 0;
    let startReadingHeight = 0;

    resizer.addEventListener('mousedown', (e) => {
        isResizing = true;
        startY = e.clientY;
        startWordsHeight = wordsPanel.offsetHeight;
        startReadingHeight = readingPanel.offsetHeight;

        resizer.classList.add('dragging');
        document.body.classList.add('resizing');
        document.body.style.cursor = 'row-resize';

        e.preventDefault();
    });

    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;

        const deltaY = e.clientY - startY;
        const totalHeight = startWordsHeight + startReadingHeight;

        // Calculate new heights with min constraints
        let newWordsHeight = Math.max(100, startWordsHeight + deltaY);
        let newReadingHeight = Math.max(100, startReadingHeight - deltaY);

        // Ensure total stays the same
        if (newWordsHeight + newReadingHeight > totalHeight) {
            if (deltaY > 0) {
                newReadingHeight = totalHeight - newWordsHeight;
            } else {
                newWordsHeight = totalHeight - newReadingHeight;
            }
        }

        wordsPanel.style.flex = `0 0 ${newWordsHeight}px`;
        readingPanel.style.flex = `0 0 ${newReadingHeight}px`;
    });

    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove('dragging');
            document.body.classList.remove('resizing');
            document.body.style.cursor = '';
        }
    });
}

// Entity detection and redaction functions
async function runEntityDetection() {
    if (!state.lastExtractionInfo?.csvPath) return;

    try {
        const response = await fetch('/api/redact', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                csv_path: state.lastExtractionInfo.csvPath,
                detect_names: true,
                detect_dates: true,
                detect_codes: true,
            }),
        });

        const data = await response.json();

        if (data.success) {
            // Update words with entity info
            state.words = data.words || state.words;
            state.entityStats = {
                total: data.redacted_count || 0,
                names: data.by_type?.NAME || 0,
                dates: data.by_type?.DATE || 0,
                codes: data.by_type?.CODE || 0,
            };
            updateRedactionSection();
            console.log(`Entity detection: ${state.entityStats.total} words marked for redaction`);
        }
    } catch (error) {
        console.error('Entity detection failed:', error);
    }
}

function updateRedactionSection() {
    const section = document.getElementById('redaction-section');
    if (state.entityStats.total > 0) {
        section.classList.remove('hidden');
        document.getElementById('redaction-total').textContent = state.entityStats.total;
        document.getElementById('stat-names').textContent = state.entityStats.names;
        document.getElementById('stat-dates').textContent = state.entityStats.dates;
        document.getElementById('stat-codes').textContent = state.entityStats.codes;
    } else {
        section.classList.add('hidden');
    }
}

function hideRedactionSection() {
    document.getElementById('redaction-section')?.classList.add('hidden');
    state.entityStats = { total: 0, names: 0, dates: 0, codes: 0 };
}

async function applyRedactions() {
    if (!state.currentPdfPath || !state.lastExtractionInfo?.csvPath) {
        alert('No PDF or extraction available');
        return;
    }

    try {
        const btn = document.getElementById('apply-redactions-btn');
        btn.disabled = true;
        btn.textContent = 'Applying...';

        const response = await fetch('/api/apply', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                pdf_path: state.currentPdfPath,
                csv_path: state.lastExtractionInfo.csvPath,
            }),
        });

        if (response.ok) {
            // Download the redacted PDF
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = state.currentPdfPath.replace('.pdf', '_redacted.pdf').split('/').pop();
            a.click();
            URL.revokeObjectURL(url);
        } else {
            const data = await response.json();
            alert('Failed to apply redactions: ' + (data.detail || 'Unknown error'));
        }
    } catch (error) {
        console.error('Apply redactions failed:', error);
        alert('Failed to apply redactions: ' + error.message);
    } finally {
        const btn = document.getElementById('apply-redactions-btn');
        btn.disabled = false;
        btn.textContent = '■ Apply Redactions';
    }
}

// Show file info modal with CSV preview
function showFileInfoModal() {
    const modal = document.getElementById('file-info-modal');
    const info = state.lastExtractionInfo;

    if (!info) {
        alert('No extraction info available');
        return;
    }

    // Populate info fields
    document.getElementById('info-pdf-path').textContent = info.pdfPath;
    document.getElementById('info-csv-path').textContent = info.csvPath;
    document.getElementById('info-word-count').textContent = info.wordCount;

    // Populate CSV preview from current words data
    const thead = document.getElementById('csv-preview-head');
    const tbody = document.getElementById('csv-preview-body');

    // Clear previous content
    thead.innerHTML = '';
    tbody.innerHTML = '';

    if (state.words.length > 0) {
        // Create header row
        const headers = ['ID', 'Text', 'Source', 'Status', 'Conf', 'x0', 'y0', 'x1', 'y1'];
        const headerRow = document.createElement('tr');
        headers.forEach(h => {
            const th = document.createElement('th');
            th.textContent = h;
            headerRow.appendChild(th);
        });
        thead.appendChild(headerRow);

        // Create data rows (limit to first 50 for performance)
        const maxRows = Math.min(state.words.length, 50);
        for (let i = 0; i < maxRows; i++) {
            const word = state.words[i];
            const row = document.createElement('tr');
            const cells = [
                word.word_id,
                word.text,
                word.source,
                word.status,
                (word.confidence || 0).toFixed(1),
                (word.x0 || 0).toFixed(1),
                (word.y0 || 0).toFixed(1),
                (word.x1 || 0).toFixed(1),
                (word.y1 || 0).toFixed(1),
            ];
            cells.forEach(cell => {
                const td = document.createElement('td');
                td.textContent = cell;
                row.appendChild(td);
            });
            tbody.appendChild(row);
        }

        if (state.words.length > maxRows) {
            const row = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = headers.length;
            td.style.textAlign = 'center';
            td.style.fontStyle = 'italic';
            td.textContent = `... and ${state.words.length - maxRows} more rows`;
            row.appendChild(td);
            tbody.appendChild(row);
        }
    }

    modal.classList.remove('hidden');
}
