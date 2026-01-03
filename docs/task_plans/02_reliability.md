````markdown name=tasks.md
# Fetcher / Fetcher-ETL Reliability Tasks (Agent-Executable, Low-Bloat)

**Date:** 2026-01-03  
**Repo:** `grahama1970/fetcher`  
**Entrypoints:**  
- Consumer CLI: `fetcher` → `fetcher.cli:app`  
- ETL CLI alias: `fetcher-etl` → `fetcher.workflows.fetcher:main`

## Project goals (must satisfy)
1. **Install reliability:** clean installs work on any machine/CI; no local path dependencies.
2. **Consumer “just works”:** best-effort persists downloads whenever any payload exists (even when content is junk).
3. **Straightforward diagnosis:** stable provenance fields + opt-in ETL metrics output.
4. **No bloat/brittleness:** minimal new knobs, minimal new concepts, additive-only schema changes, test-backed.

## Non-goals (explicitly out of scope)
- Adding arbitrary user-configurable fallback ordering.
- Adding new heavy dependencies or new logging frameworks.
- Breaking changes to existing JSON outputs.
- Expanding consumer CLI flags beyond what exists today.

---

# Global constraints / guardrails (apply to every PR)
1. **Additive schema only:** never rename existing keys; only add optional keys.
2. **Single writer rule for provenance:** only the network layer (`web_fetch.py`) writes provenance fields.
3. **No default extra output:** new human-readable metrics must be opt-in (flag gated), not printed by default.
4. **Complexity cap:** avoid new configuration matrices. Prefer fixed defaults + one or two safe toggles.
5. **Test every behavior change:** each PR must add/adjust tests proving the intended invariant.

---

# PR plan (strict order; do not reorder)

## PR-001 (P0) — Make `fetcher[alternates]` installable via GitHub (packaging-only)
### Problem
`pyproject.toml` currently references `scillm` via a local `file:///...` path, which is not installable on other machines.

### Required change
Update `pyproject.toml` optional dependency:

- From:
  - `scillm @ file:///home/graham/workspace/experiments/litellm`
- To (deterministic pin required):
  - `scillm @ git+https://github.com/grahama1970/scillm.git@<PIN>`

**PIN rule:** prefer a commit SHA. If branch pin is temporarily necessary, it must be replaced with SHA within 1–2 follow-up commits.

### Files to modify
- `pyproject.toml`

### Acceptance criteria
- On a clean machine: `uv add "fetcher[alternates]"` succeeds.
- `python -c "import scillm"` succeeds with extras installed.
- No runtime behavior changes in fetcher/etl.
- `fetcher[alternates]` remains pinned to a specific `scillm` GitHub SHA for reproducibility; update the pin in `pyproject.toml` when you want newer SciLLM behavior (default branch is `main`).

### Tests / smokes
- Add a minimal installation/import smoke as a script under `scripts/` or as a CI step (no new deps).

---

## PR-002 (P0) — Consumer hard invariant: always persist payload bytes before any gating
### Problem
Consumer CLI must “always download”. ETL content gating can clear `raw_bytes` / `text` for junk verdicts; if consumer persists late, it may lose the payload.

### Hard invariant (must implement)
> If the network fetch returned any payload bytes (or any meaningful body), the consumer run must persist **exactly those bytes** into `downloads/` once, regardless of content verdict.

### Required implementation approach (avoid ambiguity)
1. **Determine the earliest point in the consumer flow where the original payload exists.**
   - Consumer must persist downloads immediately after fetch returns results, **before** calling `evaluate_result_content` (or any function that can clear bytes/text).
2. Use existing helper: `fetcher.workflows.download_utils.persist_downloads(...)` with `allow_junk=True`.
3. Record the persisted artifact path into the consumer item summary (`download_path`) even for junk results when bytes exist.
4. If persistence fails:
   - Add an error entry (string) to the item `errors[]`.
   - Continue to produce `consumer_summary.json` + `Walkthrough.md`.
   - Exit code behavior remains governed by existing `--soft-fail` logic.

### Files to modify
- `src/fetcher/consumer.py` (consumer runner orchestration)
- Possibly `src/fetcher/workflows/download_utils.py` ONLY if needed to expose a stable return value or metadata key. Prefer not to change it unless necessary.

### Tests to add (must be deterministic; no network)
Add unit tests that:
1. **Junk-but-bytes case:**
   - Create a `FetchResult` representing a paywall/junk scenario *with* `raw_bytes` set (or `text` set), and ensure consumer persistence writes a file and sets `download_path`.
2. **No-payload case:**
   - A result with no `raw_bytes` and no `text` should not create a download; summary includes a clear warning like `download_missing_no_payload` (or equivalent).
3. **Byte integrity case (strongly recommended):**
   - Persisted file contents hash equals the pre-gate bytes hash.

### Acceptance criteria
- Consumer produces a download artifact whenever bytes exist, even if `content_verdict != ok`.
- Consumer summary accurately reflects `download_path` for those items.
- Tests pass without network/Playwright.

---

## PR-003 (P0) — Standardize provenance: add `fetch_path` + `fallback_reason` (small enum)
### Problem
Fallback complexity is hidden but hard to diagnose. Need stable, minimal provenance.

### Required outputs (additive metadata; stable names)
- `metadata["fetch_path"]`: list of strings showing **successful** mechanisms used, e.g.
  - `["aiohttp"]`
  - `["aiohttp", "playwright"]`
  - `["aiohttp", "proxy_rotation"]`
  - `["aiohttp", "wayback"]`
  - `["aiohttp", "wayback", "jina"]` (only if it truly succeeds in that order)
- `metadata["fallback_reason"]`: string or null

### Strict constraints to prevent brittleness
1. **Small reason enum:** initial set MUST be limited to ~6–8 values max.
   - Proposed initial codes (edit as needed, but keep small):
     - `rate_limited_429`
     - `bot_blocked`
     - `cloudflare_block_page`
     - `soft_404`
     - `empty_payload`
     - `http_error_retry_exhausted`
2. **No “attempted steps” list in v1.** Only record successful path.
3. **Single writer rule:** only `web_fetch.py` sets/updates these keys.

### Files to modify
- `src/fetcher/workflows/web_fetch.py`
- `src/fetcher/consumer.py` (optional: include these in consumer summary item, additive only)
- `src/fetcher/workflows/fetcher.py` (optional: ensure provenance not overwritten)

### Tests to add (deterministic; no network)
- Monkeypatch-based unit tests verifying `fetch_path` updates when:
  - Playwright path is selected
  - Wayback path is used
  - Jina path is used
  - Proxy rotation is used (already tested elsewhere; extend assertion to provenance)

### Acceptance criteria
- Every result has `fetch_path` (at least `["aiohttp"]`).
- When a fallback is used, `fallback_reason` is populated with one of the allowed enum values.
- Provenance is stable and consistent across consumer and ETL.

---

## PR-004 (P1) — ETL: opt-in metrics output (stable schema; O(n) computation)
### Problem
ETL metrics exist in artifacts/audit but are not surfaced cleanly. Diagnosis requires opening files.

### Required CLI additions (ETL only; do not affect consumer)
- `--metrics-json <path|->`: emit stable JSON metrics object
- `--print-metrics`: print a compact human summary to **stderr** (not stdout)

### Metrics schema (must match exactly)
```json
{
  "counts": {"total": 0, "success": 0, "failed": 0},
  "content_verdict_counts": {},
  "paywall_verdict_counts": {},
  "fallback_counts": {},
  "proxy_rotation": {"attempted": 0, "success": 0},
  "rate_limit_metrics": null
}
```

**Definitions:**
- `counts.total`: number of results
- `counts.success`: `status == 200` AND has usable payload (use existing `has_text_payload` logic)
- `counts.failed`: `status != 200` OR missing payload
- `content_verdict_counts`: count by `metadata["content_verdict"]` (string)
- `paywall_verdict_counts`: count by `metadata["paywall_verdict"]` (string)
- `fallback_counts`: count each fallback/provider name found in `fetch_path` excluding the initial `aiohttp` if you want provider-only counts; pick one approach and document it.
- `proxy_rotation`: counts derived from result metadata flags (e.g., `proxy_rotation_used`) or from audit if present.
- `rate_limit_metrics`: if present from `URLFetcher.fetch_many()` audit, pass through; else null.

### Implementation constraints
- Metrics computation must be O(n) over results with constant work per result.
- No new deps.
- No default printing unless flags are used.

### Files to modify
- `src/fetcher/workflows/fetcher.py` (`main()` and/or `run_fetch_pipeline()`)

### Tests to add
- Unit test that builds a small list of `FetchResult` stubs with metadata and verifies:
  - schema keys exist
  - counts match expected values
  - `--metrics-json -` emits valid JSON
- Do not rely on network.

### Acceptance criteria
- `fetcher-etl --inventory ... --metrics-json -` prints correct JSON to stdout.
- `--print-metrics` prints a short summary to stderr.
- Default ETL behavior unchanged.

---

## PR-005 (P1) — Resolver eligibility refinement (ETL-only; hard-signal driven)
### Problem
Resolver is currently heavily tied to `paywall_domains`. Some bot blocks / interstitials may not be resolved even when clearly blocked.

### Guardrails (prevent surprise behavior)
- **ETL-only** in this iteration. Consumer remains minimal.
- Only broaden resolver eligibility when there is a **hard signal** from provenance:
  - `fallback_reason in {"bot_blocked","cloudflare_block_page"}` OR
  - status in `{401,403,429,451,503}` (existing behavior)
- Do NOT broaden based on “empty payload” alone (too many false positives).
- Preserve existing resolver limit/caps (`limit` parameter) strictly.

### Files to modify
- `src/fetcher/workflows/paywall_utils.py`

### Tests to add
- Unit test: entry with `status==200`, `fallback_reason="bot_blocked"`, no text payload becomes resolver-candidate even when domain not in paywall_domains.
- Ensure no paid resolvers are called unless explicitly enabled (existing gating must remain intact).

### Acceptance criteria
- Resolver triggers improve for hard-signal blocks.
- No increase in resolver usage for non-blocked normal pages.
- Deterministic outcomes and tests pass.

---

# Deferred tasks (explicitly NOT required for initial reliability milestone)
## D-001 — Consolidate Brave logic into one provider wrapper
**Reason deferred:** valuable but not required to achieve install reliability + consumer guarantee + provenance + metrics. Consolidation introduces churn/merge conflicts.

## D-002 — Fallback order configurability / strategy presets
**Reason deferred:** high risk of brittleness and support-matrix complexity. Reconsider only after provenance + metrics show a real need.

---

# Regression checklist (must run each PR)
1. `uv run pytest`
2. `uv run fetcher --help` and `uv run fetcher --help-full`
3. `uv run fetcher-etl --help`
4. `scripts/smoke_fetcher_cli.sh` (where network is available; otherwise keep unit tests comprehensive)
5. Verify outputs:
   - Consumer: `consumer_summary.json` + `Walkthrough.md` + `downloads/`
   - ETL: `.results.jsonl` + `.audit.json` + run artifacts

---

# Definition of Done (reliability milestone)
All P0 + P1 PRs (001–005) merged, and:
- `fetcher[alternates]` installs on clean machines.
- Consumer always persists payload bytes when any exist.
- Every result carries stable provenance (`fetch_path`, `fallback_reason`).
- ETL can emit stable metrics via opt-in flags.
- Resolver handles hard-signal bot blocks more reliably without broad heuristics.
````
