# [CLAUDE] MCP Server - Extract Tool

> **Type:** Claude (interactive session)
> **Status:** ready
> **Priority:** P2 (medium)
> **Created:** 2026-01-29

## Goal

Build an MCP server that exposes Portadoc's PDF extraction via `extract(pdf_path)` tool, returning CSV data. This allows Claude Code and other MCP clients to use Portadoc for OCR.

## Context

- Portadoc has a working `extract_words()` function in `src/portadoc/extractor.py`
- Output formatting via `write_harmonized_csv()` in `src/portadoc/output.py`
- Using fastmcp for simpler API
- Server will use stdio transport (standard for Claude Code)
- No CLI options exposed - just defaults from `config/harmonize.yaml`

## Task

### Phase 1: Setup & Dependencies

1. Add `fastmcp` to project dependencies
2. Create `src/portadoc/mcp/__init__.py`
3. Create `src/portadoc/mcp/server.py` with basic fastmcp structure

### Phase 2: Implement Extract Tool

1. Implement `extract` tool that:
   - Takes `pdf_path: str` (absolute path on disk)
   - Calls `extract_words()` with defaults
   - Formats result as CSV string via `write_harmonized_csv()`
   - Returns CSV string
2. Add basic error handling (file not found, invalid PDF)

### Phase 3: CLI Entry Point

1. Add `mcp` command to CLI: `./portadoc mcp` starts the MCP server
2. Update `pyproject.toml` if needed for entry point

### Phase 4: Testing

1. Unit tests for extract tool logic (mock MCP protocol)
2. Integration smoke test that actually starts server and calls tool
3. Manual test: add to Claude Code and extract from test PDF

## Architecture

### Files to Create

| File | Purpose |
|------|---------|
| `src/portadoc/mcp/__init__.py` | Package init |
| `src/portadoc/mcp/server.py` | MCP server with extract tool |
| `tests/test_mcp.py` | Unit and integration tests |

### Files to Modify

| File | Changes |
|------|---------|
| `pyproject.toml` | Add fastmcp dependency |
| `src/portadoc/cli.py` | Add `mcp` command |

### Extract Tool Interface

```python
@mcp.tool()
def extract(pdf_path: str) -> str:
    """
    Extract words from a PDF using multi-engine OCR.

    Args:
        pdf_path: Absolute path to PDF file on disk

    Returns:
        CSV string with columns: word_id, page, x0, y0, x1, y1, text, status, source, conf, rotation, ...
    """
```

### Server Structure

```python
# src/portadoc/mcp/server.py
from fastmcp import FastMCP
from io import StringIO
from pathlib import Path

from ..extractor import extract_words
from ..output import write_harmonized_csv

mcp = FastMCP("portadoc")

@mcp.tool()
def extract(pdf_path: str) -> str:
    """Extract words from PDF, return CSV."""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    words = extract_words(path)

    output = StringIO()
    write_harmonized_csv(words, output)
    return output.getvalue()
```

## Constraints

- Use fastmcp, not raw mcp SDK
- stdio transport only (no SSE/HTTP)
- Defaults only - no configurable options on the tool
- PDF must exist on disk (no base64)
- Return full CSV with all columns (let client filter if needed)

## Acceptance

- [ ] `./portadoc mcp` starts MCP server without errors
- [ ] Server responds to tool list request showing `extract` tool
- [ ] `extract("/path/to/peter_lou.pdf")` returns valid CSV string
- [ ] CSV contains expected columns and data
- [ ] Unit tests pass: `pytest tests/test_mcp.py -v`
- [ ] Integration test passes: server starts, tool executes, returns CSV
- [ ] Manual test: Claude Code can call the tool successfully

## Validation Commands

```bash
# Unit tests
pytest tests/test_mcp.py -v

# Manual server test (run in terminal, Ctrl+C to stop)
./portadoc mcp

# Test with mcp-cli if available
echo '{"method": "tools/list"}' | ./portadoc mcp
```

## Files to Reference

- `src/portadoc/extractor.py` - `extract_words()` function
- `src/portadoc/output.py` - `write_harmonized_csv()` function
- `src/portadoc/cli.py` - CLI structure for adding `mcp` command
- `docs/tickets/CLAUDE-001-kraken-integration.md` - Example ticket structure

## Test Data

- `data/input/peter_lou.pdf` - Clean 3-page test PDF (use for testing)
- Expected: ~400 words extracted

---

**Session Notes:** (Claude fills this during the session)
