# Documentation

## Structure

```
docs/
├── tickets/           # Active work
│   ├── TEMPLATE_RALPH.md   # Template for Ralph (autonomous) tickets
│   ├── TEMPLATE_CLAUDE.md  # Template for Claude (interactive) tickets
│   ├── archive/       # Completed tickets
│   └── freezer/       # Paused/deprioritized tickets
└── archive/           # Historical docs from previous work
    └── round7/        # Round 7 development files
```

## Ticket Types

### [RALPH] Tickets
For autonomous agent loops. These tickets:
- Span multiple sessions
- Have explicit entry/exit conditions
- Include validation commands and metrics
- Track progress with checklists

### [CLAUDE] Tickets
For interactive Claude sessions. These tickets:
- Complete in single session
- More conversational
- May need clarification
- Good for exploration, debugging, one-off fixes

## Creating Tickets

1. Copy the appropriate template
2. Name it: `RALPH-NNN-short-description.md` or `CLAUDE-NNN-short-description.md`
3. Fill in all sections
4. Move to `archive/` when complete, `freezer/` if paused
