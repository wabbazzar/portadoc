# Fix Plan - Browser-Based Portadoc Client

## Phase 1: Project Setup
- [x] Create `src/portadoc/browser/` directory structure
- [x] Set up TypeScript build with Vite
- [x] Create `index.html` with basic layout
- [x] Add npm dependencies: pdfjs-dist, tesseract.js

## Phase 2: PDF Loading
- [x] Implement `pdf-loader.ts` with PDF.js
- [x] Convert PDF pages to canvas/ImageData
- [x] Add page navigation UI
- [x] Handle multi-page PDFs

## Phase 3: Tesseract.js Integration
- [x] Implement `ocr/tesseract.ts` wrapper
- [x] Load Tesseract worker and language data
- [x] Extract words with bounding boxes
- [x] Add progress callback for UI

## Phase 4: docTR-TFJS Integration
- [ ] Clone/adapt models from doctr-tfjs-demo
- [ ] Implement `ocr/doctr.ts` wrapper
- [ ] Load detection model (db_mobilenet_v2)
- [ ] Load recognition model (crnn_vgg16_bn)
- [ ] Extract words with bounding boxes
- [ ] Add progress callback for UI

## Phase 5: Reading Order and Harmonization
- [x] Port `Word` and `BBox` interfaces to `models.ts`
- [x] Port geometric clustering algorithm from `src/portadoc/geometric_clustering.py`:
  - [x] Union-Find data structure
  - [x] `calculate_distance_thresholds()` - Q1-based threshold calculation
  - [x] `detect_column_boundaries()` - row-based gap analysis
  - [x] `build_clusters()` - spatial proximity clustering
  - [x] `sort_words_within_cluster()` - y-fuzz row grouping
  - [x] `group_clusters_into_row_bands()` - cluster ordering
  - [x] `order_words_by_reading()` - main entry point
- [ ] Port `find_word_match()` to TypeScript (for harmonization - deferred until docTR)
- [ ] Port `harmonize_words()` to TypeScript (for harmonization - deferred until docTR)
- [ ] Port deduplication logic (for harmonization - deferred until docTR)

## Phase 6: UI and Export
- [x] Display extracted words with bbox overlays
- [x] Add word list panel (like existing web UI)
- [x] Implement CSV export
- [x] Implement JSON export
- [x] Add model loading progress indicators

## Phase 7: Validation and Polish
- [x] Benchmark: peter_lou.pdf with Tesseract.js only → verify ≥80% word match ✓ (90.9%)
- [ ] Benchmark: peter_lou.pdf with docTR-TFJS only → verify ≥80% word match (deferred)
- [x] Verify reading order matches ground truth CSV sequence ✓
- [x] Stress test: peter_lou_50dpi.pdf processes without crashing ✓
- [x] Add error handling for model load failures
- [x] Screenshot validation with dev-browser ✓

## Completed
- Phase 1: Project Setup (directory, Vite, HTML, dependencies)
- Phase 2: PDF Loading (pdf-loader.ts with PDF.js)
- Phase 3: Tesseract.js Integration (ocr/tesseract.ts wrapper)
- Phase 5: Geometric clustering algorithm ported (geometric-clustering.ts)
- Phase 6: UI and export (bbox overlays, word list, CSV/JSON export)
- Phase 7: Tesseract.js validation PASSED (90.9% word match)

## Validation Results

### Tesseract.js Benchmark (peter_lou.pdf - clean)
| Page | Ground Truth | Browser | Matches | Match % |
|------|--------------|---------|---------|---------|
| 1    | 86 words     | 90      | 72      | 83.7%   |
| 2    | 160 words    | 165     | 151     | 94.4%   |
| 3    | 104 words    | 109     | 95      | 91.3%   |
| **Total** | **350 words** | **364** | **318** | **90.9%** |

**Target: ≥80% → PASS**

### Stress Test (peter_lou_50dpi.pdf - degraded)
- Page 1: 92 words extracted
- Page 2: 184 words extracted
- Page 3: 122 words extracted
- **Result: No crashes, all pages processed successfully**

## Notes
- docTR-TFJS integration deferred - requires downloading and hosting TensorFlow.js models (~50-100MB)
- Tesseract.js alone achieves 90.9% word match, exceeding the 80% target
- Reading order is working correctly (geometric clustering algorithm ported)
- Harmonization logic deferred until docTR is integrated
