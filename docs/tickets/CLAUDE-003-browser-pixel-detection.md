# [CLAUDE] Port Pixel Detection to Browser

> **Type:** Claude (interactive session)
> **Status:** complete
> **Priority:** P2 (medium)
> **Created:** 2026-01-26

## Goal

Port the Python pixel detection module (`src/portadoc/detection.py`) to TypeScript for the browser client, enabling detection of OCR-missed content without server-side Python dependencies.

## Context

The browser client (`src/portadoc/browser/`) currently supports Tesseract.js and docTR for OCR but lacks the pixel-based detection capabilities that the Python backend provides. The Python version uses OpenCV (cv2) and NumPy for:

- Detecting logo/image regions missed by OCR
- Finding horizontal lines (separators, underlines)
- Finding vertical lines (margins, borders)

The browser version needs equivalent functionality using browser-native APIs (Canvas API, typed arrays) without external image processing libraries.

## Task

Port all functions from `src/portadoc/detection.py` to a new TypeScript module:

1. **Create `src/portadoc/browser/detection.ts`** with these functions:
   - `bboxOverlapsAny()` - Check bbox overlap with existing boxes
   - `detectLogoRegions()` - Detect large boxy regions (logos, images)
   - `detectHorizontalLines()` - Detect wide thin shapes
   - `detectVerticalLines()` - Detect tall thin shapes
   - `detectMissedContent()` - Main entry point combining all detection

2. **Implement image processing using Canvas API**:
   - Grayscale conversion via pixel manipulation
   - Binary thresholding (manual implementation)
   - Morphological operations (dilate, open) via kernel convolution
   - Contour detection (connected component labeling)

3. **Maintain algorithm parity**:
   - Same filtering thresholds (min_area, min_dimension, aspect ratios)
   - Same overlap detection logic (IoU + center check)
   - Same coordinate scaling (image pixels → PDF points)

4. **Integrate with existing browser models**:
   - Use `BBox` and `Word` types from `models.ts`
   - Return `Word[]` with `engine: "pixel_detector"`

## Constraints

- **No external image processing libraries** (no opencv.js, no jimp)
- Pure TypeScript with Canvas API and typed arrays only
- Must work in all modern browsers (Chrome, Firefox, Safari, Edge)
- Keep performance reasonable for images up to 4000x4000 pixels
- Match Python behavior as closely as possible

## Acceptance

- [x] `detection.ts` exists with all 5 functions ported
- [x] Grayscale + binary threshold implemented via Canvas
- [x] Morphological dilate operation working
- [x] Morphological open operation working (for line detection)
- [x] Connected component / contour finding implemented
- [x] Logo region detection matches Python output
- [x] Horizontal line detection matches Python output
- [x] Vertical line detection matches Python output
- [x] Integration point ready for browser OCR pipeline
- [x] Manual testing on sample PDF shows similar detection results

## Files to Reference

- `src/portadoc/detection.py` - Python source to port (299 lines)
- `src/portadoc/browser/models.ts` - TypeScript types to use
- `src/portadoc/browser/geometric-clustering.ts` - Example of browser image processing patterns
- `src/portadoc/browser/ocr/tesseract.ts` - Example of async image handling

## Algorithm Notes

### Morphological Operations

The Python version uses OpenCV's morphological operations. Browser equivalents:

**Dilation**: Expand white regions using a structuring element
```
For each pixel:
  If any neighbor in kernel is white, output is white
```

**Opening**: Erosion followed by dilation (removes small noise)
```
eroded = erode(image, kernel)
opened = dilate(eroded, kernel)
```

### Contour Detection

OpenCV's `findContours` with `RETR_EXTERNAL` finds outer boundaries of connected white regions. Browser equivalent options:

1. **Connected Component Labeling** (flood-fill based)
2. **Moore-Neighbor Tracing** (boundary following)
3. **Simple bounding box extraction** from labeled regions

For this use case, we only need bounding boxes, not actual contour points, so connected component labeling with bbox extraction is sufficient.

### Threshold Values (from Python)

```typescript
// Logo detection
const MIN_AREA_PTS = 1000;        // Minimum area in PDF points²
const MIN_DIMENSION_PTS = 30;     // Minimum width/height in pts
const MAX_ASPECT_RATIO = 20;      // Skip line-like shapes

// Horizontal lines
const MIN_WIDTH_PTS = 100;        // Minimum line width
const MAX_HEIGHT_PTS = 10;        // Maximum line height
const MIN_H_ASPECT = 10;          // Width/height ratio

// Vertical lines
const MIN_HEIGHT_PTS = 50;        // Minimum line height
const MAX_WIDTH_PTS = 10;         // Maximum line width
const MIN_V_ASPECT = 5;           // Height/width ratio

// Overlap detection
const OVERLAP_THRESHOLD = 0.1;    // IoU threshold
```

---

**Session Notes:**

### Implementation Complete (2026-01-26)

**Files created/modified:**
- `src/portadoc/browser/detection.ts` - New 450-line module with all detection functions
- `src/portadoc/browser/app.ts` - Integrated pixel detection into extraction pipeline
- `src/portadoc/browser/index.html` - Added PX checkbox for pixel detection toggle
- `src/portadoc/browser/styles.css` - Added orange styling for pixel_detector engine

**Image processing primitives implemented (pure TypeScript, no libraries):**
- `toGrayscale()` - RGBA to grayscale using luminance formula
- `binaryThresholdInv()` - Inverse binary threshold (matches cv2.THRESH_BINARY_INV)
- `dilate()` - Morphological dilation with rectangular kernel
- `erode()` - Morphological erosion with rectangular kernel
- `morphOpen()` - Opening operation (erode then dilate)
- `findContourBoundingBoxes()` - Connected component labeling via flood-fill

**Detection functions ported:**
- `bboxOverlapsAny()` - IoU + center-inside overlap check
- `detectLogoRegions()` - Large boxy regions (logos, images)
- `detectHorizontalLines()` - Wide thin shapes (separators)
- `detectVerticalLines()` - Tall thin shapes (borders)
- `detectMissedContent()` - Main entry combining all detection

**Sanity check:**
- Added `runSanityCheck()` function with expected bbox for peter_lou_50dpi.pdf
- Expected logo bbox: (102, 117) - (189, 220) in canvas pixels
- Test passes with 0.0 px distance

**Tested on peter_lou_50dpi.pdf page 1:**
- 4 pixel regions detected:
  1. Paw logo at top-left header
  2. Horizontal line under header
  3. Cat emoji in patient info
  4. Additional decorative element
- All regions correctly excluded from OCR word overlap

