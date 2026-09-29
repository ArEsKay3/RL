---
name: feedback-full-paths
description: "User wants every output file named with its complete absolute path in the reply, never a bare filename plus \"in the same directory\""
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7cb50873-4185-4239-a055-62bd9f83c0ac
  modified: 2026-09-24T22:39:25.567Z
---

Always give every output file (SVG, PDF, txt, csv) as its complete absolute path in the final message, one per line. Never say "same directory" or list bare filenames under a directory header.

**Why:** rkirby clicks the paths straight from the terminal; a bare filename cannot be clicked (2026-09-24: "I always need complete paths so I can click them").

**How to apply:** when reporting artifacts, write the full `/scratch/.../file.svg` path for each file, even when several share a directory. Related: [[feedback-arm-labels]].
