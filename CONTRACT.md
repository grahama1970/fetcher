# Fetcher Module Contract

This document defines the expectations, stages, and verification steps for the
shared fetcher toolkit (`fetcher.workflows`). Every repo that depends on the
module must satisfy this contract to ensure consistent behavior across Sparta,
SciLLM, and Extractor.

## 1. Purpose

Fetcher is responsible for deterministic retrieval, normalization, and
classification of remote artifacts (HTML, PDF, JSON, binaries) so downstream
automation can reason about the content. Specifically, fetcher MUST:

- Fetch a URL (HTTP, Playwright, Brave, alternates) while honoring policy knobs.
- Annotate each result with paywall verdict, content verdict, and provenance
  metadata before any artifacts are persisted.
- Persist blobs/rolling windows only when the content verdict is `ok`.
- Surface unresolved URLs in `run/artifacts/outstanding_domains_summary.json`.

## 2. Processing Stages

1. **Prefilter** (`prefilters.evaluate_body_prefilter`)
   - Normalize URL, deduplicate inventory entries, decide whether the entry is
     a link hub (fan-out) or a terminal document.

2. **Fetch** (`web_fetch.URLFetcher`)
   - Perform HTTP fetch via aiohttp.
   - If HTML is incomplete or JS heavy, automatically retry with Playwright.
   - For GitHub blobs, rewrite to raw.githubusercontent.com and fall back to the
     `gh` CLI if necessary.
   - For blocked responses, retry via Wayback or r.jina.ai.

3. **Content Quality Gate** (`extract_utils.evaluate_result_content`)
   - Run trafilatura first, readability-lxml second.
   - Compute `content_verdict`, length, link density, paywall marker hits.
   - If verdict != `ok`, mark the result as failed (no blob persists) and include
     excerpt + reasons in metadata.

4. **Paywall Detection** (`download_utils.annotate_paywall_metadata`)
   - Skip detection for safe domains/suffixes, otherwise run
     `paywall_detector.detect_paywall` and attach verdict/score/indicators.

5. **Download Modes / Externalization** (`download_utils.apply_download_mode`)
   - `text`: inline text only.
   - `download_only`: persist blob (PDF/HTML/etc.) with checksum metadata.
   - `rolling_extract`: persist blob plus JSONL rolling windows.
   - Post-write verification (`verify_blob_content`) re-checks saved bytes and
     updates metadata if corruption is detected.

6. **Alternates & Resolving** (`paywall_utils.resolve_paywalled_entries`)
   - If paywall verdict ∈ {maybe, likely} and policy allows, attempt Brave
     alternates or overrides before giving up.

7. **Outstanding Reporting** (`outstanding_utils.build_outstanding_reports`)
   - Emit classic reports plus `outstanding_domains_summary.json` with
     categories: `paywall`, `needs_login_or_playwright`, `content_thin`,
     `content_link_hub`, `broken_or_moved`, `retry`, `needs_whitelist`.

## 3. Failure Rules

- **Paywall / Junk content**: result is FAILED, alternates may run, URL appears
  in outstanding summary with the relevant category.
- **Link hub / TOC**: success only if link-hub detection is intended; otherwise
  fan-out handles it and the hub itself is suppressed.
- **Playwright unreachable**: treat as failure with `method="error"` and
  include the stack trace in metadata for triage.
- **GitHub CLI failure**: revert to HTTP error; do not emit partial blobs.

## 4. Sanity Test (Required)

Use the representative sample at `docs/fetcher_sample_urls.md` to verify every
change to fetcher. At minimum, the following command MUST pass before merging:

```bash
PYTHONPATH=src \
uv run python -m fetcher.workflows.fetcher \
  --inventory docs/fetcher_sample_urls.jsonl \
  --run-artifacts run/artifacts/contract_sanity
```

Acceptance criteria for this test:

1. `run/artifacts/contract_sanity/outstanding_domains_summary.json` contains
   only genuine failures (login-required, CAC gated, or 404s). No D3FEND/ATT&CK
   technique page should appear as `broken_or_moved` or `link_hub`.
2. All `status == 200` rows in `docs/fetcher_sample_urls.results.jsonl` MUST have
   `content_verdict == "ok"` for narrative pages (D3FEND techniques, CNAS, etc.).
   Non-article hubs may remain `link_hub` only if the page truly lacks prose and
   is intended as a TOC.
3. GitHub rewrites (blob -> raw) include `github_fetch_url` metadata and the
   saved blob exists on disk.
4. No result with `content_verdict != "ok"` persists blobs or rolling windows.

## 5. Required Unit Tests

All future changes must keep the following tests green:

- `tests/test_fetcher_helpers.py`
- `tests/test_github_utils.py`
- `tests/test_extract_utils.py`
- `tests/test_outstanding_utils.py`
- Any new tests covering regression fixes (e.g., SPA detection, alternates).

## 6. Logging & Artifacts

- Every `FetchResult` JSON line MUST include:
  - `method` (aiohttp, playwright, pdf, wayback, github_cli, etc.).
  - `paywall_verdict`, `content_verdict`, `content_text_len`,
    `content_link_density`, `content_marker_hits`, `content_reasons`.
- `run/artifacts/<run-id>/` MUST contain:
  - `outstanding_domains_summary.json`
  - `outstanding_urls_summary.json`
  - `outstanding_controls_remaining.json`
  - `alternate_urls_applied.jsonl` (if alternates executed)

## 7. Status & Hang Reporting

- Long-running fetches must emit periodic progress logs (every N URLs) so a
  human can see that the run is alive. If a run exceeds 5 minutes without new
  logs, cancel and report the last stage.
- When a run is cancelled or times out, always record a short status in the CLI
  output (e.g., "aborted after 319s while waiting for Playwright").

Maintaining this contract ensures fetcher remains reliable across all agents and
that humans can quickly reason about failures when they do occur.
