# [RALPH] Ticket Title

> **Type:** Ralph (autonomous agent)
> **Status:** draft | ready | in-progress | blocked | complete
> **Priority:** P0 (critical) | P1 (high) | P2 (medium) | P3 (low)
> **Created:** YYYY-MM-DD
> **Updated:** YYYY-MM-DD

## Objective

One-sentence summary of what this ticket accomplishes.

## Context

Background information Ralph needs:
- Why this matters
- What led to this ticket
- Any relevant prior work

---

## Ralph Setup Checklist

Before starting Ralph, ensure these files exist in the project root:

### Required Files

- [ ] **PROMPT.md** - Ralph's instructions for this ticket (see template below)
- [ ] **@fix_plan.md** - Task checklist with `[ ]` / `[x]` items
- [ ] **@AGENT.md** - Build, test, and run instructions
- [ ] **specs/** directory - Specifications Ralph references

### Optional but Recommended

- [ ] **.claude/settings.json** - Sandbox permissions (allows uninterrupted work)
- [ ] **screenshots/** directory - For GUI verification (if using dev-browser)

---

## Scope

### In Scope
- Specific deliverable 1
- Specific deliverable 2

### Out of Scope
- Thing to explicitly avoid
- Future work NOT part of this ticket

---

## Task Breakdown (@fix_plan.md content)

Copy this to `@fix_plan.md`:

```markdown
# Fix Plan - [Ticket Title]

## High Priority
- [ ] Task 1: Description
- [ ] Task 2: Description

## Medium Priority
- [ ] Task 3: Description
- [ ] Task 4: Description

## Completed
(Ralph moves items here as they finish)

## Notes
- Key constraint or consideration
```

---

## PROMPT.md Template

Copy and customize for `PROMPT.md`:

```markdown
# Ralph Development Instructions - [Ticket Title]

## Context
You are Ralph, an autonomous AI agent working on **[Project Name]**.
[1-2 sentences about the project]

## Session Objective
[Clear statement of what this Ralph session should accomplish]

## CRITICAL: Environment Setup
**ALWAYS activate venv before Python commands:**
\`\`\`bash
source .venv/bin/activate
\`\`\`

## Test Data
- `path/to/test/file` - Description
- `path/to/ground/truth` - Description

## Task Summary
[Brief description of the main task]

### The Problem
[What's wrong or what needs to be built]

### The Solution
[High-level approach]

### Key Files
- `src/file.py` - What to modify
- `config/file.yaml` - What to configure

### Validation
\`\`\`bash
# Command to verify success
make test
# Expected: All tests pass
\`\`\`

## Key Principles
- Complete tasks in order from @fix_plan.md
- Test after each change
- Update @fix_plan.md with [x] when done
- Keep changes minimal and focused

## Status Reporting (CRITICAL)

At the end of EVERY response, include:

\`\`\`
---RALPH_STATUS---
STATUS: IN_PROGRESS | COMPLETE | BLOCKED
TASKS_COMPLETED_THIS_LOOP: <number>
FILES_MODIFIED: <number>
TESTS_STATUS: PASSING | FAILING | NOT_RUN
WORK_TYPE: IMPLEMENTATION | TESTING | DOCUMENTATION | REFACTORING | DEBUGGING
EXIT_SIGNAL: false | true
RECOMMENDATION: <one line summary of next action>
---END_RALPH_STATUS---
\`\`\`

### EXIT_SIGNAL = true when:
1. All items in @fix_plan.md are marked [x]
2. All validation commands pass
3. No regressions introduced

## Quick Start
1. Read @fix_plan.md thoroughly
2. Start with first unchecked item
3. Test after each change
4. Update @fix_plan.md with [x] when done
```

---

## Exit Conditions

All must be true for Ralph to exit successfully:

- [ ] All @fix_plan.md items marked `[x]`
- [ ] Validation command passes: `[specific command]`
- [ ] No regressions: `[specific check]`
- [ ] Metric achieved: `[e.g., "F1 >= 95%"]`

---

## Validation Commands

```bash
# Primary validation - what Ralph runs to verify success
command here

# Regression check
command here
```

---

## GUI Testing (if applicable)

If this ticket involves UI work, Ralph can use **dev-browser** for visual verification:

```
Use the dev-browser skill to:
1. Navigate to http://localhost:8000
2. Take screenshot of [specific element]
3. Verify [expected behavior]
Save screenshots to screenshots/[task-name].png
```

---

## What Makes a Good Ralph Task

### Good Tasks
- **Atomic**: Can complete in 1-3 loop iterations
- **Measurable**: Has clear pass/fail validation command
- **Specific**: Exact files and changes identified
- **Testable**: Can verify without human judgment

### Bad Tasks
- "Improve performance" (too vague)
- "Make it better" (no clear exit)
- "Research options" (no deliverable)
- Tasks requiring human decisions mid-work

### Example Good Task
```markdown
- [ ] Add `text_match_bonus` parameter to `find_word_match()` in `src/harmonize.py`
      Validation: `pytest tests/test_harmonize.py -k text_match`
```

### Example Bad Task
```markdown
- [ ] Improve the harmonization algorithm
      (No specific changes, no validation command)
```

---

## Progress Log

Ralph updates this as work progresses:

```
[YYYY-MM-DD HH:MM] Loop 1: Started task 1, modified src/file.py
[YYYY-MM-DD HH:MM] Loop 2: Completed task 1, tests passing
[YYYY-MM-DD HH:MM] Loop 3: Started task 2...
```

---

## Results

Final metrics and outcomes (filled on completion):

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Metric 1 | X | Y | Z |

---

## Completion Notes

Summary of what was done and any follow-up tickets needed.
