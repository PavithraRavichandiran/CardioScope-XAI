---
name: feedback-collab-style
description: How the user wants to collaborate — Claude builds, user commits after review
metadata:
  type: feedback
---

Claude does all the coding work phase by phase, step by step. When a phase is complete, Claude tells the user to commit with a specific commit message. The user reviews and commits themselves — Claude never commits directly.

**Why:** User wants a clean, reviewable git history with meaningful commit messages per phase.

**How to apply:** After completing each phase or logical milestone, output a clear "Ready to commit" block with the exact commit message to use. Never run git commit yourself.
