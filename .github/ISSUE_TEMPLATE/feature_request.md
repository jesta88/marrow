---
name: Feature request
about: Suggest an addition or change
title: ''
labels: enhancement
---

**Problem**
What are you trying to do that marrow doesn't support today?

**Proposed solution**
What you'd like to see. Note if it touches the public API (`marrow.h`) or the `.mrw` wire format.

**Alternatives**
Other approaches you considered.

**Scope check**
marrow is intentionally small: a zero-dependency, batch-first skeletal + baked-crowd runtime that emits
skinning palettes and leaves scheduling, GPU upload, and the animation state machine to the caller. It
does not grow GPU hierarchy evaluation, IK, masks, or runtime blend-graphs (that's the Tier-B
guardrail). Does your request fit that scope?
