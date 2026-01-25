# Agent Instructions - Browser Portadoc Client

## Build Commands

```bash
# Initial setup (run once)
cd src/portadoc/browser
npm install

# Development server with hot reload
npm run dev
# Opens at http://localhost:5173

# Production build
npm run build

# Type checking
npm run typecheck
```

## Test Commands

```bash
# No automated tests yet - use manual validation with dev-browser

# Validation workflow:
# 1. Start dev server: cd src/portadoc/browser && npm run dev
# 2. Use dev-browser skill to navigate to http://localhost:5173
# 3. Upload test PDFs and verify extraction
```

## Run Commands

```bash
# Development
cd src/portadoc/browser && npm run dev

# Preview production build
cd src/portadoc/browser && npm run preview
```

## Test Data

| File | Description |
|------|-------------|
| `data/input/peter_lou.pdf` | Clean 3-page test PDF |
| `data/input/peter_lou_50dpi.pdf` | Degraded version for stress testing |
| `data/input/peter_lou_words_slim.csv` | Ground truth (401 words, correct reading order) |
| `data/input/peter_lou_50dpi_page*.jpg` | Pre-converted JPG images for quick testing |

## Validation Criteria

### Benchmark: peter_lou.pdf (clean)
- Tesseract.js alone: ≥80% word match (≥320 of 401 words)
- docTR-TFJS alone: ≥80% word match (≥320 of 401 words)
- Reading order: must match ground truth CSV sequence

### Stress test: peter_lou_50dpi.pdf (degraded)
- Must process without crashing
- Words extracted (accuracy not required)

## Architecture

```
src/portadoc/browser/
├── index.html              # Main page
├── app.ts                  # Main application logic
├── pdf-loader.ts           # PDF.js wrapper
├── ocr/
│   ├── tesseract.ts        # Tesseract.js wrapper
│   └── doctr.ts            # docTR-TFJS wrapper
├── geometric-clustering.ts # Reading order algorithm
├── harmonize.ts            # Multi-engine result fusion
├── models.ts               # TypeScript interfaces
└── styles.css              # Styling
```

## Key Dependencies

```json
{
  "dependencies": {
    "pdfjs-dist": "^4.0.0",
    "tesseract.js": "^5.0.0",
    "@tensorflow/tfjs": "^4.0.0"
  },
  "devDependencies": {
    "typescript": "^5.0.0",
    "vite": "^5.0.0"
  }
}
```

## GUI Testing

Use **dev-browser** skill for visual verification:

```
1. Navigate to http://localhost:5173
2. Upload PDF or image
3. Wait for OCR to complete
4. Take screenshot
5. Verify bounding boxes are displayed
Save screenshots to screenshots/browser-client-[step].png
```

## Common Issues

1. **CORS errors with PDF.js worker**: Use local worker file or configure Vite proxy
2. **TensorFlow.js memory**: Call `tf.dispose()` on tensors after use
3. **Tesseract.js slow first load**: Language data downloads ~10MB on first use
4. **docTR model loading**: Models are ~50-100MB, show progress indicator
