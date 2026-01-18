# Portadoc

PDF word extraction for document redaction. Extracts text with word-level bounding boxes using multi-engine OCR (Tesseract, EasyOCR, PaddleOCR, docTR, Surya) with smart harmonization.

## For Agents

**Before you do anything:** Use `./portadoc` from repo root (auto-uses venv) or `source .venv/bin/activate`

### What type of work is this?

| Type | Where to Look |
|------|---------------|
| **Ralph ticket (autonomous)** | Check `docs/tickets/` for RALPH-* ticket, set up required files |
| **Claude ticket (interactive)** | Check `docs/tickets/` for CLAUDE-* ticket |
| **Quick fix or exploration** | This file has what you need |

### Tickets

Active tickets live in `docs/tickets/`. Two types:

- **[RALPH]** - Autonomous agent loops. Requires setup files (see below).
- **[CLAUDE]** - Interactive sessions. Single session, conversational.

Templates: `docs/tickets/TEMPLATE_RALPH.md`, `docs/tickets/TEMPLATE_CLAUDE.md`

Completed: `docs/tickets/archive/`
Paused: `docs/tickets/freezer/`

---

## Ralph System

Ralph runs autonomous development loops. Each iteration gets fresh context, works through `@fix_plan.md`, and reports status.

### Required Files for Ralph

Before starting a Ralph session, these must exist in project root:

| File | Purpose |
|------|---------|
| `PROMPT.md` | Instructions for Ralph (objectives, validation, status format) |
| `@fix_plan.md` | Task checklist with `[ ]` / `[x]` items |
| `@AGENT.md` | Build, test, run instructions |
| `specs/` | Specifications Ralph references |

Optional: `.claude/settings.json` (sandbox permissions), `screenshots/` (GUI verification)

### Ralph Status Block

Ralph must output this at end of every response:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | DEBUGGING
EXIT_SIGNAL: false | true
RECOMMENDATION: <next action>
---END_RALPH_STATUS---
```

**EXIT_SIGNAL = true** when all `@fix_plan.md` items are `[x]` and validation passes.

### Good Ralph Tasks

- **Atomic**: Completable in 1-3 loops
- **Measurable**: Has validation command with pass/fail
- **Specific**: Names exact files and changes
- **Testable**: No human judgment needed

Bad: "Improve performance" / Good: "Add caching to `get_words()` in `src/api.py`, validation: `pytest -k cache`"

### Running Ralph

```bash
~/.ralph/ralph_loop.sh    # Start loop
~/.ralph/ralph_monitor.sh # Monitor progress
```

---

## GUI Testing with dev-browser

For UI work, use the **dev-browser** skill for visual verification:

```
Use dev-browser to:
1. Navigate to http://localhost:8000
2. Take screenshot
3. Verify expected behavior
```

Screenshots go in `screenshots/` for documentation.

---

## Quick Reference

### Run Stuff
```bash
./portadoc --help            # CLI help (no venv activation needed)
make help                    # See all Makefile commands
make check                   # Verify OCR engines
make serve                   # Web UI at localhost:8000
```

### CLI Commands

```bash
# Extract words from PDF
./portadoc extract input.pdf                       # All engines (default)
./portadoc extract input.pdf -o output.csv         # Save to file
./portadoc extract input.pdf --primary surya       # Use Surya as primary
./portadoc extract input.pdf --no-surya            # Disable Surya

# Evaluate against ground truth
./portadoc eval input.pdf ground_truth.csv
./portadoc eval input.pdf gt.csv -v                # Verbose mode

# Check OCR availability
./portadoc check

# Web UI
./portadoc serve                                    # http://localhost:8000
./portadoc serve --reload                           # Dev mode with auto-reload
```

### Key Flags (extract & eval)

| Flag | Default | Description |
|------|---------|-------------|
| `--primary ENGINE` | tesseract | Primary engine for bbox authority (tesseract/surya/doctr/easyocr/paddleocr) |
| `--no-surya` | enabled | Disable Surya OCR |
| `--no-easyocr` | enabled | Disable EasyOCR |
| `--no-doctr` | enabled | Disable docTR |
| `--no-paddleocr` | disabled | Disable PaddleOCR |
| `--preprocess LEVEL` | auto | none/light/standard/aggressive/auto |
| `--psm N` | 3 | Tesseract page segmentation mode (0-13) |
| `--upscale N` | none | Super-resolution (2 or 4) |
| `--config PATH` | config/harmonize.yaml | Custom config file |
| `-o PATH` | stdout | Output file path |
| `--format FORMAT` | csv | Output format (csv/json) |
| `--progress` | off | Show progress bar |

### Config-Driven Primary Engine

Set in `config/harmonize.yaml`:
```yaml
harmonize:
  primary:
    engine: tesseract  # or surya, doctr
    weight: 1.0
```

CLI `--primary` overrides config.

---

## Architecture

```
src/portadoc/
├── cli.py              # CLI commands (click)
├── extractor.py        # Main extraction pipeline
├── harmonize.py        # Multi-engine result fusion
├── config.py           # YAML config loader
├── models.py           # Data classes (Word, BBox, etc.)
├── metrics.py          # F1, precision, recall
├── preprocess.py       # OpenCV image enhancement
├── detection.py        # Pixel fallback detection
├── pdf.py              # PyMuPDF PDF handling
├── api.py              # FastAPI REST endpoints
├── ocr/
│   ├── tesseract.py    # Default primary (word-level)
│   ├── surya_ocr.py    # SOTA accuracy (word-level, polygon)
│   ├── doctr_ocr.py    # High accuracy (word-level)
│   ├── easyocr.py      # Degraded docs (line-level)
│   └── paddleocr.py    # General (word-level)
└── web/
    ├── app.py          # Web UI routes
    └── static/         # Frontend (JS, CSS, HTML)

config/harmonize.yaml   # OCR and harmonization settings
data/input/             # Test PDFs and ground truth
```

### Extraction Pipeline
```
PDF → Image → [Preprocess] → [Super-res] → Multi-OCR → Harmonize → Output
```

1. **PDF to Image**: PyMuPDF renders pages
2. **Preprocess**: Optional OpenCV enhancement
3. **Multi-OCR**: Run enabled engines (Tesseract primary)
4. **Harmonize**: Smart bbox matching, text voting, deduplication
5. **Output**: CSV with word_id, text, bbox, status, source

### Engine Characteristics
| Engine | BBox Type | Best For | Notes |
|--------|-----------|----------|-------|
| **Tesseract** | word-level | Default primary, best F1 | Reliable bbox precision |
| **Surya** | word-level (polygon) | SOTA text accuracy | Polygon→axis-aligned conversion |
| **docTR** | word-level | High text accuracy | Good secondary |
| **EasyOCR** | line-level | Degraded docs | Needs word decomposition |
| **PaddleOCR** | word-level | General | Has ~10px bbox offset |

---

## Test Data

- `data/input/peter_lou.pdf` - Clean 3-page test PDF
- `data/input/peter_lou_50dpi.pdf` - Degraded version
- `data/input/peter_lou_words_slim.csv` - Ground truth (401 words)

## Current Performance

| Config | Clean F1 | Degraded F1 | Notes |
|--------|----------|-------------|-------|
| Tesseract-only | 99.00% | 81.55% | Best bbox precision |
| All 4 engines | 95.92% | 74.11% | Better text accuracy |

---

## Git Practices

### Commit Strategy (MANDATORY)

**Commit after EVERY meaningful change - NO EXCEPTIONS**

1. **Test First**: Always verify changes work before committing
2. **Commit Immediately**: After each phase/feature completion
3. **Break Large Changes**: If a task takes >30 min, break into sub-commits
4. **Clean Messages**: NO "Co-Authored-By: Claude" or emoji in commits

### Commit Message Standards

Follow Conventional Commits format:

```
<type>(<scope>): <subject under 50 chars>

<body (optional)>
- Brief explanation of what changed
- Why this change was needed
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Scopes** (common for portadoc):
`ocr`, `harmonize`, `cli`, `web`, `config`, `models`, `metrics`, `tests`, `pipeline`

**Examples**:
```
feat(ocr): add Surya engine with polygon bbox support
fix(harmonize): resolve duplicate word detection threshold
refactor(cli): consolidate extraction flags
test(metrics): add F1 score boundary tests
chore(deps): update PyMuPDF to 1.24.0
```

### Recovery Protocol

If anything breaks:
```bash
git log --oneline -5           # See recent commits
git reset --hard <commit>      # Rollback to working state
git diff HEAD~1               # See what changed in last commit
```

### Git Commands Reference

```bash
git status                    # Current state
git diff                      # Unstaged changes
git diff --staged             # Staged changes
git log --oneline -10         # Recent history
git add -p                    # Interactive staging
git stash                     # Save work temporarily
git stash pop                 # Restore stashed work
```

---

## Dev Guidelines

### Always
- Activate venv before Python: `source .venv/bin/activate`
- Update Makefile when adding CLI commands
- Test on both clean and degraded PDFs
- Keep changes minimal and focused
- **Commit after completing each task**

### Never
- Commit to main without testing
- Add GPU/CUDA dependencies (CPU-only)
- Over-engineer or add unnecessary features
- **Add "Co-Authored-By: Claude" to commits**

### Makefile Convention
```makefile
##@ Category

target: ## Description
	$(PORTADOC) command --flags
```

---

## Files at a Glance

| File | Purpose |
|------|---------|
| `CLAUDE.md` | This file - agent springboard |
| `Makefile` | All commands, single source of truth |
| `config/harmonize.yaml` | OCR settings and thresholds |
| `specs/portadoc.md` | Full specification |
| `docs/tickets/` | Active work tickets |
| `docs/tickets/TEMPLATE_RALPH.md` | How to set up Ralph tickets |
