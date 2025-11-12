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

## 5. Optional alternates
Install the extras and set SciLLM env vars:
```bash
uv add "fetcher[alternates]"
uv run python -m fetcher.workflows.fetcher --use-alternates inventory.jsonl
```

## 6. Tests + lint
```bash
uv run pytest
uv run ruff check src
```
