# Quickstart

Follow these steps to spin up the standalone Fetcher project.

## 1. Prerequisites
- Python 3.12.10 (match the repo's `.venv` for reproducibility)
- [uv](https://docs.astral.sh/uv/) for env + lock management (`pip install uv` or use the official installer)
- Playwright system deps (`sudo npx playwright install-deps` on Ubuntu if needed)

## 2. Clone + install
```bash
cd /home/graham/workspace/experiments
cd fetcher
uv venv --python=3.12.10 .venv
source .venv/bin/activate
cd ..
uv sync --project fetcher
uv run playwright install --with-deps chromium
```
`uv venv` pins the local virtual environment to CPython 3.12.10 (matching production). `uv sync` then reads `pyproject.toml` + `uv.lock`, installing dependencies into that `.venv`. The final command fetches the Chromium bundle required for SPA fallbacks.

## 3. Configure secrets
Update `.env` (already tracked for convenience):
```
BRAVE_API_KEY=sk-your-brave-key
CHUTES_API_BASE=https://llm.chutes.ai/v1
CHUTES_API_KEY=sk-your-chutes-key
CHUTES_TEXT_MODEL=chutes/gpt-4o
```
If you skip SciLLM alternates you can leave the `CHUTES_*` values blank.

## 3b. Doctor + dry-run (recommended)
Validate your environment and inputs before running larger jobs:
```bash
fetcher doctor
fetcher get --dry-run https://example.com
fetcher-etl --dry-run --inventory docs/smokes/urls.jsonl
```
`fetcher doctor` prints a redacted environment report, while `--dry-run` validates inputs and policy without fetching. The smoke inventory contains a small set of stable URLs (HTML, PDF, GitHub blob, 404) for quick ETL checks.

## 4. Run your first fetch
```bash
uv run python - <<'PY'
import asyncio
from pathlib import Path
from fetcher.workflows.web_fetch import URLFetcher, FetchConfig, write_results

async def main():
    config = FetchConfig(concurrency=4, per_domain=2)
    fetcher = URLFetcher(config)
    entries = [{"url": "https://www.nasa.gov"}]
    results, audit = await fetcher.fetch_many(entries)
    write_results(results, Path("artifacts/nasa.jsonl"))
    print(audit)

asyncio.run(main())
PY
```
(You can also wire `FetcherPolicy` + `generate_alternate_urls` for production pipelines.)

Notes:
- `fetcher.workflows.fetcher.fetch_url(...)` disables link-hub fan-out by default so single-URL calls return quickly.
- The CLI supports `--no-fanout` to disable fan-out for a specific run.
- Fetch runs now also emit `extracted_text_path` (when `FETCHER_EMIT_EXTRACTED_TEXT=1`, default) and a `junk_table.md` summary under the run artifacts directory for easy triage.
- Optional: emit LLM-friendly markdown with `FETCHER_EMIT_MARKDOWN=1` (writes `markdown_path` and `run/artifacts/.../markdown/*.md`).
- Optional: emit pruned “fit markdown” with `FETCHER_EMIT_FIT_MARKDOWN=1` (default) **when markdown is enabled** (writes `fit_markdown_path` and `run/artifacts/.../fit_markdown/*.md`).

### 4b. Consumer CLI (best-effort)
The consumer CLI provides a minimal, surf-style interface with fewer knobs and
deterministic outputs. It always attempts to persist downloads, even if content
is judged junk.

```bash
fetcher get https://www.nasa.gov
fetcher get-manifest urls.txt
fetcher get-manifest - < urls.txt
fetcher --help-full
fetcher --find markdown
```

Outputs:
- `consumer_summary.json` (stable summary)
- `Walkthrough.md` (deterministic rendering of the summary)

Use `fetcher-etl` when you need full ETL reports and knobs.
Missing Brave/Playwright dependencies surface as environment warnings in stderr and the summary/audit.
Use `--soft-fail` on the consumer CLI to keep exit code 0 when warnings exist.
ETL (`fetcher-etl`) supports `--soft-fail` for the same behavior.

### ETL metrics (opt-in)
```bash
fetcher-etl --inventory urls.jsonl --metrics-json metrics.json
fetcher-etl --inventory urls.jsonl --metrics-json - --print-metrics
```

`--metrics-json -` writes the metrics object to stdout (and suppresses the default summary).

## 5. Optional alternates
Install the extras and set SciLLM env vars:
```bash
uv add "fetcher[alternates]"
uv run python -m fetcher.workflows.fetcher --use-alternates inventory.jsonl
```
Note: the `alternates` extra is pinned to a specific `scillm` GitHub SHA for reproducibility. Update the pin in `pyproject.toml` if you want newer SciLLM behavior (the default branch is `main`).

## 6. Download modes (optional)
Decide how you want bodies stored before running large jobs:

```bash
export FETCHER_DOWNLOAD_MODE=rolling_extract
export FETCHER_ROLLING_WINDOW_SIZE=6000
export FETCHER_ROLLING_WINDOW_STEP=3000
export FETCHER_SINGLE_RUN_ARTIFACTS=run/fetcher_artifacts
```

`download_only` writes every body to `run/artifacts/downloads/sha.ext`, while `rolling_extract` also materializes overlapping text windows under `run/artifacts/rolling_windows/` for streaming ingestion. Use `text` to keep legacy inline behavior.

Sentence segmentation prefers spaCy's sentencizer when installed and falls back to a regex-based splitter otherwise.

## 7. Optional PDF password cracking
- Install helper: `uv add pdferli` (or `pip install pdferli` inside the venv).
- Enable and tune via env:

```
export FETCHER_PDF_CRACK_ENABLE=1
export FETCHER_PDF_CRACK_CHARSET=0123456789
export FETCHER_PDF_CRACK_MINLEN=4
export FETCHER_PDF_CRACK_MAXLEN=6
export FETCHER_PDF_CRACK_TIMEOUT=15   # seconds, default 15
export FETCHER_PDF_CRACK_PROCESSES=2  # CPU cores to use
```

When a PDF requires a password, fetcher will attempt a bounded brute-force; on success text is extracted, on failure the entry is marked `content_verdict="password_protected"` and skipped downstream.

## 8. Optional IP rotation fallback (Step 06)

Step 06 teams that already maintain IPRoyal credentials (see `memory/scripts/smokes/iproyal_login.py`) can let fetcher retry rate-limited domains through the proxy automatically. Configure the env before launching the run:

```
export SPARTA_STEP06_PROXY_HOST=gw.iproyal.com
export SPARTA_STEP06_PROXY_PORT=12321
export SPARTA_STEP06_PROXY_USER=team
export SPARTA_STEP06_PROXY_PASSWORD=super-secret
# Optional tuning
export SPARTA_STEP06_PROXY_DOMAINS=d3fend.mitre.org,atlas.mitre.org
export SPARTA_STEP06_PROXY_STATUSES=429,403
export SPARTA_STEP06_PROXY_HINTS="rate limit,too many requests"
```

- You can pass a full URI via `SPARTA_STEP06_PROXY_URL=http://user:pass@gw.iproyal.com:12321` or `SPARTA_STEP06_PROXY_CREDENTIALS=user:pass`.
- When the `SPARTA_STEP06_PROXY_*` vars are missing, fetcher falls back to `IPROYAL_HOST|PORT|USER|PASSWORD`, keeping the proxy smokes and Step 06 runner in sync.
- Allowlisted domains default to `d3fend.mitre.org` (set `SPARTA_STEP06_PROXY_DOMAINS_DISABLE_DEFAULTS=1` to opt out). Hints and status codes are comma-separated lists.
- Disable the feature at runtime with `SPARTA_STEP06_PROXY_DISABLE=1`.

When a throttled response (e.g., 429) matches the allowlist, the fetcher retries through the proxy before Wayback/Jina. Each proxied `FetchResult` advertises `proxy_rotation_*` metadata, and the audit JSON includes a `proxy_rotation` section summarizing attempts, successes, and per-domain usage so Step 06 operators can trace every rotation.

## 9. Refresh brittle mirrors (optional)

Keep local overrides deterministic by mirroring high-churn URLs into `src/fetcher/data/local_sources/`:

```
uv run python -m fetcher.tools.mirror_refresher \
  --manifest src/fetcher/data/mirror_sources.json \
  --out src/fetcher/data/local_sources
```

This CLI reads the manifest, fetches each URL, and writes the byte-for-byte copy under `local_sources/` so overrides can reference a stable `file://` path.

## 10. Tests + lint
```bash
uv run pytest
uv run ruff check src
```
