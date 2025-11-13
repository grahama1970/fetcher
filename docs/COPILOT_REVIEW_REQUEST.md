# Refactor fetcher helpers, GitHub transport, and outstanding reporting

## Repository and branch

- **Repo:** `grahama1970/fetcher`
- **Branch:** `main`
- **Paths of interest:**
  - `src/fetcher/workflows/web_fetch.py`
  - `src/fetcher/workflows/download_utils.py`
  - `src/fetcher/workflows/fetcher.py`
  - `src/fetcher/workflows/fetcher_config.py`
  - `src/fetcher/workflows/fetcher_utils.py`
  - `src/fetcher/workflows/github_utils.py`
  - `src/fetcher/workflows/outstanding_utils.py`
  - `tests/test_fetcher_helpers.py`
  - `tests/test_github_utils.py`
  - `tests/test_outstanding_utils.py`

## Summary

Fetcher now needs to serve Sparta, SciLLM, and LiteLLM, so this patch:

1. Moves remaining helper logic out of `fetcher.py`, adds GitHub transport support (raw + `gh api` fallback), and expands Playwright/sign-in handling for .gov/.mil and JS-heavy news sites.
2. Introduces policy-driven safe domains/suffixes so `.gov`/`.mil` and vetted domains stop triggering paywall heuritics.
3. Emits a human-friendly `outstanding_domains_summary.json` categorizing every failed URL (whitelist candidate, paywall, login/Playwright, broken, retry) to keep humans in the loop.
4. Adds regression coverage for the new helpers (GitHub request rewriting, outstanding categorization, sentence splitter sanity).

Please review the orchestration boundaries, new helpers, and the policy additions.

## Objectives

1. **Helper refactor + imports** — Ensure `fetcher.py` only orchestrates (helpers live in `fetcher_utils`, `download_utils`, `outstanding_utils`).
2. **GitHub support** — `web_fetch` should rewrite `github.com/*/blob` URLs to `raw.githubusercontent.com`, honor `GITHUB_TOKEN`, and fall back to `gh api ...` when HTTP fails.
3. **Playwright gating** — Confirm the expanded `SPA_FALLBACK_DOMAINS`/`SIGNIN_MODAL_DOMAINS` are reasonable and that Playwright is triggered only when necessary.
4. **Safe domains/suffixes** — Verify `.gov/.mil` handling (via `PAYWALL_SAFE_SUFFIXES`) doesn’t regress existing paywall detection logic.
5. **Outstanding summary JSON** — Ensure `outstanding_utils.generate_outstanding_summary` writes the new JSON with accurate categorization and that tests cover the logic.
6. **Tests** — Validate the new helper tests provide meaningful coverage, especially around GitHub rewriting and outstanding categories.

## Constraints for the patch

- No vendoring of large artifacts; helpers should stay under `src/fetcher/workflows/`.
- Keep `URLFetcher` backwards compatible (existing CLI and integrators should not break).
- GitHub CLI fallback must be optional (gated by `FETCHER_ENABLE_GH` and availability of `gh`).
- Outstanding summary JSON should live alongside the legacy summaries under `run/artifacts/`.

## Acceptance criteria

- `PYTHONPATH=src uv run pytest tests/test_fetcher_helpers.py tests/test_github_utils.py tests/test_outstanding_utils.py` passes.
- Running `PYTHONPATH=src uv run python -m fetcher.workflows.fetcher --inventory <inventory>` produces `run/artifacts/outstanding_domains_summary.json` with categorized entries.
- Fetching a `github.com/.../blob/...` URL stores metadata indicating the raw URL and (when used) the `gh` CLI path.
- Expanded Playwright domains render LinkedIn/CISA/etc. without leaving URLs in the outstanding list once credentials are available.

## Test plan

1. `PYTHONPATH=src uv run pytest tests/test_fetcher_helpers.py tests/test_github_utils.py tests/test_outstanding_utils.py`
2. `PYTHONPATH=src uv run python -m fetcher.workflows.fetcher --url https://github.com/org/repo/blob/main/README.md --run-artifacts run/artifacts/manual`
3. `PYTHONPATH=src uv run python -m fetcher.workflows.fetcher --inventory /home/graham/workspace/experiments/devops/sparta/data/processed/url_inventory.jsonl --run-artifacts run/artifacts/latest`

## Clarifying questions

1. Are there any external consumers still depending on helpers inside `fetcher.py`, or is it safe that everything routes through the new utility modules?
2. Do we need additional throttling or caching around the `gh` CLI fallback to avoid rate limits during bulk fetches?
3. Does `PAYWALL_SAFE_SUFFIXES = (".gov", ".mil")` align with your compliance requirements, or should we keep a narrower whitelist?
4. **Key request:** From Copilot’s perspective, is fetcher duplicating functionality already provided by a well-maintained Python module (e.g., `newspaper3k`, `trafilatura`, `playwright-stealth` wrappers)? If so, please call that out explicitly so we can evaluate adopting it instead of maintaining bespoke code.

## Deliverable

Please reply with a standard Copilot Code Review covering:

- High-level assessment of correctness, regressions, and testing gaps.
- Detailed findings (bugs, risky assumptions, missing validation) with file/line references.
- Answers to the clarifying questions above (especially whether we’re reinventing existing libraries).

Use the formatting from `docs/COPILOT_REVIEW_REQUEST_EXAMPLE.md` for tone/structure.
