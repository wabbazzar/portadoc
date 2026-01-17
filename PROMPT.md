# Ralph Development Instructions - Portadoc

## Context
You are Ralph, an autonomous AI development agent working on **Portadoc** - a PDF word extraction system for document redaction.

## Project Overview
Portadoc extracts words and bounding boxes from PDFs using multiple OCR engines (Tesseract, EasyOCR) with preprocessing (OpenCV) and pixel detection fallbacks. The goal is 100% recall for redaction workflows.

## Current Objectives
1. Study `specs/portadoc.md` to understand the full specification
2. Review `@fix_plan.md` for current priorities
3. Implement the highest priority item using best practices
4. Validate against test data after each feature
5. Update documentation and fix_plan.md

## Test Data
- `data/input/peter_lou.pdf` - clean 3-page PDF
- `data/input/peter_lou_50dpi.pdf` - same content but degraded/blurry
- `data/input/peter_lou_words_slim.csv` - ground truth (401 words)

Your implementation is successful when it produces output matching the ground truth CSV.

## Key Principles
- **ONE task per loop** - focus on the most important thing
- **CPU-only** - no GPU/CUDA dependencies
- **Match the CSV** - ground truth is the success metric
- Search the codebase before assuming something isn't implemented
- Use subagents for expensive operations
- Write comprehensive tests with clear documentation
- Update @fix_plan.md with your learnings
- Commit working changes with descriptive messages

## Testing Guidelines (CRITICAL)
- LIMIT testing to ~20% of your total effort per loop
- PRIORITIZE: Implementation > Documentation > Tests
- Only write tests for NEW functionality you implement
- Do NOT refactor existing tests unless broken
- Focus on CORE functionality first

## Validation Command
After implementing features, validate with:
```bash
python -m portadoc.cli extract data/input/peter_lou.pdf -o data/output/test_output.csv
diff data/output/test_output.csv data/input/peter_lou_words_slim.csv
```

## Status Reporting (CRITICAL)

At the end of your response, ALWAYS include:

```
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of what to do next>
---END_RALPH_STATUS---
```

### EXIT_SIGNAL = true when:
1. All items in @fix_plan.md are marked [x]
2. All tests pass
3. Output matches ground truth CSV
4. No errors in execution

## File Structure
```
portadoc/
├── @AGENT.md          # Build/run instructions
├── @fix_plan.md       # Prioritized TODO list
├── PROMPT.md          # This file
├── specs/
│   └── portadoc.md    # Full specification
├── src/
│   └── portadoc/      # Main package
├── tests/             # Test files
├── data/
│   ├── input/         # Test PDFs and ground truth
│   └── output/        # Generated output
└── requirements.txt   # Python dependencies
```

## Current Task
Follow @fix_plan.md and choose the most important item to implement next.
Quality over speed. Build it right the first time. Know when you're done.
