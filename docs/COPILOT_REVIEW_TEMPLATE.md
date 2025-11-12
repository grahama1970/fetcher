# Generalized Copilot Request — Patch + Answers (No PRs, No Links)

**AGENT INSTRUCTIONS**
Remember to commit and push the current branch to the repo first
Do NOT IGNORE!!!!!!!


**Project**

* Fork/Repo: `grahama1970/devops`
* Branch: `fix/restore-pipeline-steps-20251031-073204`
* Path: `git@github.com:grahama1970/devops.git#fix/restore-pipeline-steps-20251031-073204`

**Task**

* Wire Step 07 to the centralized body prefilter, log structured skip reasons, and update audits/docs so operators can pass audits before Step 08.

**Context (brief, optional)**

* Step 07 currently launches SciLLM fan-out without calling `sparta/workflows/prefilters.evaluate_body_prefilter`, leaving the Local-First policy unenforced.
* Operators need per-slice telemetry (`prefilter_reason`, `section_pref_fallback`) baked into `07_generate_knowledge.json` and 07c so “pass all audits before Step 09” remains actionable.

**Review Scope (relative paths)**

* Primary:

  * `sparta/pipeline/07b_generate_llm_knowledge_chunks.py`
  * `sparta/workflows/prefilters.py`
  * `sparta/pipeline/07c_audit.py`
  * `sparta/pipeline/CONTRACT.md`
* Also check (if needed):

  * `sparta/workflows/knowledge_chunks.py`
  * `docs/guides/CHUTES_CALLS.md`

**Objectives**

* Call `evaluate_body_prefilter(...)` on each candidate before invoking `generate_knowledge_chunks`, respecting `decision/reason/fallback` without hitting the LLM when disallowed.
* Stream per-source prefilter metadata into `knowledge_chunks.jsonl`/skip files and surface aggregate counters (`prefilter_summary`) inside `07_generate_knowledge.json`.
* Extend 07c plus CONTRACT copy so audits verify the new counters and remind operators how to remediate prefilter failures.

**Constraints**

* **Unified diff only**, inline inside a single fenced block.
* **No PRs, no hosted links, no URLs, no extra commentary.**
* Include a **one-line commit subject** inside the patch.
* **Numeric hunk headers only** (`@@ -old,+new @@`), no symbolic headers.
* Patch must apply cleanly on branch `fix/restore-pipeline-steps-20251031-073204`.
* Preserve plan→execute semantics; avoid destructive defaults.

**Acceptance (we will validate)**

* `EXECUTE=0 GK_LIMIT=2 python sparta/pipeline/07b_generate_llm_knowledge_chunks.py` prints `prefilter_summary` in the JSON output even when dry-running.
* `EXECUTE=1 GK_LIMIT=2 SPARTA_SOFT_EXIT=1 python sparta/pipeline/07b_generate_llm_knowledge_chunks.py` writes skip rows that include `prefilter_reason`/`section_pref_fallback`, and 07c passes with the new checks.
* `python sparta/pipeline/07c_audit.py` and `python sparta/pipeline/06m_audit.py` succeed, confirming the added counters don’t break existing gates and `CONTRACT.md` documents the policy.

**Deliverables (STRICT — inline only; exactly these sections, in this order)**

1. **UNIFIED_DIFF:**

```diff
# Copilot: insert the entire diff here.
```

2. **ANSWERS:**

* `Dependencies/data sources: No new pins; reuse existing ".env" + SciLLM paved-path inputs.`
* `Schema drift: Fail fast when required columns/fields disappear; don’t silently coerce.`
* `Safety: Keep all mutations behind EXECUTE/--execute; prefilter skips remain read-only.`
* `Tests/smokes: Run Step 07 with and without EXECUTE plus 07c audit.`
* `Performance: Honor GK_CONCURRENCY, SPARTA_PER_CALL_TIMEOUT, and current retry knobs.`
* `Observability: Continue emitting pipeline_exit plus the new prefilter summary lines.`

**Clarifying Questions (answer succinctly in the ANSWERS section; if unknown, reply `TBD` + minimal dependency needed)**

* Dependencies/data sources: Do we need to pin inputs/models/versions for repeatability?
* Schema drift: Should exporters/parsers tolerate missing/renamed columns with failing smokes?
* Safety: Are all mutating paths gated behind `--execute`? Any missing guards?
* Tests/smokes: Which deterministic smokes must pass (counts > 0, report count==pairs, strict formats)?
* Performance: Any batch sizes, rate limits, or timeouts/retries to honor?
* Observability: What summary lines should the CLI print on completion?

**Output Format (must match exactly; no extra text):**
UNIFIED_DIFF:

```diff
# Copilot: insert diff here.
```

ANSWERS:

* `<bullet answers in order of the clarifying questions>`

---

## Quick “Drop-In” Mini Version

**Request:** Produce a **single unified diff** (inline) for `grahama1970/devops#fix/restore-pipeline-steps-20251031-073204` that adds the Step 07 prefilter hook, records skip metadata, and updates audits/docs accordingly.
**Scope:** `sparta/pipeline/07b_generate_llm_knowledge_chunks.py`, `sparta/workflows/prefilters.py`, `sparta/pipeline/07c_audit.py`, `sparta/pipeline/CONTRACT.md`
**Constraints:** No PRs/links; include a one-line commit subject; numeric hunk headers only; patch applies cleanly.
**Acceptance:** Step 07 dry-run/run emit the prefilter summary; skip files carry reasons; 07c audit succeeds.

**Output (exact):**
UNIFIED_DIFF:

```diff
# Copilot: insert diff here.
```

ANSWERS:

* `<answers to: deps, schema drift, safety, tests, performance, observability>`

---

## Optional Toggles (copy/paste as needed)

* **Strict JSON Mode:** “All generated configs/snippets must be strict JSON: no comments, no trailing commas, no markdown/codefences inside the JSON.”
* **Flag-First DX:** “Commands and code must use explicit flag-first configuration; no hidden env defaults.”
* **Worker/Batching Defaults:** “Default ≤3 workers; batch size 10–15; retries with exponential backoff.”
* **Determinism:** “Seeded or deterministic outputs where feasible; produce minified JSON artifacts.”
* **MBOX Variant (if you ever switch modes):** Replace the UNIFIED_DIFF block with:
  **Output (exact):**
  `MBOX:` *(paste full git-format patch series; no code fences)*

---

### Placeholder Key

* `<paths…>`: Narrow file list to focus Copilot
* `<brief objectives>` / `<Acceptance>`: What “done” looks like
