---
name: feedback-terse-code-comments
description: User rejects multi-line explanatory comment blocks in code edits; keep comments to one line and never make an edit whose only content is commentary
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7aa0a354-aa1b-4011-8140-8d3122eb7495
  modified: 2026-09-12T02:40:06.715Z
---

On 2026-09-11 the user said "Stop proposing edits that are only comments, this is garbage" after I replaced a
removed config default with a four-line comment explaining why it was removed, and had earlier added several
multi-line rationale blocks (job numbers, dates, failure narratives) inside `dolphin_convergence_common.sh`.

**Why:** the user wants the code to carry code; history and rationale belong in commit messages and in these
memory notes, not inline. Long comment blocks bloat diffs and reviews.

Followed by: "The code is not a place for you to leave yourself notes."

**How to apply:** no rationale comments in code at all in this workspace. Explain a setting only if the code
itself is genuinely unreadable without it, in one line, with no job IDs, dates, or narrative. Everything else
goes in the commit message body or in these memory files. Never make an edit whose only content is comments.
