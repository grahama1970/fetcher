# Fetcher

Fetcher is the standalone packaging of SciLLM/Sparta's battle-tested crawling stack. It turns the internal `workflows.fetcher` module into a self-service library with:

- Deterministic Brave+Wayback resolution for gated links
- Async web fetching with Playwright, MuPDF, OCR, and HTML quality prefilters
- Structured audit records (controls, worksheets, paywall hints)
- Optional SciLLM-powered alternate generation for controls that still fail validation

## Why another project?

The LiteLLM monorepo mixes provider adapters, proxy gateway, and knowledge ingestion utilities. This repo extracts only the fetcher so small tooling stacks (MCP servers, ETL tasks, dashboards) can depend on a lightweight package without importing the full gateway.

## Layout

```
fetcher/
├── pyproject.toml
├── uv.lock
├── .env
├── QUICKSTART.md
├── FEATURES.md
├── README.md
└── src/fetcher
    ├── core/keys.py
    ├── workflows/
    │   ├── alternate_urls.py
    │   ├── fetcher.py
    │   ├── fetcher_config.py
    │   ├── prefilters.py
    │   └── web_fetch.py
    └── data/
        ├── overrides.json
        ├── local_sources/
        └── processed/
```

## Data expectations

- `src/fetcher/data/overrides.json` ships the stable rules used in SciLLM pipelines.
- Drop any mirrored PDFs or TXT files into `src/fetcher/data/local_sources/` if you need deterministic replacements.
- Populate `src/fetcher/data/processed/controls_context.jsonl` with optional control metadata to enrich alternate generation.

Large crawls can keep JSONL outputs manageable by exporting `FETCHER_TEXT_INLINE_MAX_BYTES` (e.g., `FETCHER_TEXT_INLINE_MAX_BYTES=200000`). When set, oversized response bodies are written to `run/artifacts/text_blobs/` and the JSON metadata stores a pointer via `text_path`.

## Download modes & rolling windows

Use `FETCHER_DOWNLOAD_MODE` to control how response bodies are stored:

- `text` (default): inline text in JSON unless `FETCHER_TEXT_INLINE_MAX_BYTES` externalizes it.
- `download_only`: persist every body (binary or text) to `run/artifacts/downloads/<sha>.<ext>` and zero the inline `text` field.
- `rolling_extract`: keep the download + emit JSONL rolling windows under `run/artifacts/rolling_windows/` using `FETCHER_ROLLING_WINDOW_SIZE`, `FETCHER_ROLLING_WINDOW_STEP`, and optional `FETCHER_ROLLING_WINDOW_MAX_WINDOWS`. Windows follow spaCy sentence boundaries when spaCy is installed (otherwise fall back to a regex splitter) so no chunk cuts off a sentence mid-way.

Each HTML result also records a `paywall_detection` blob (score, verdict, indicators) derived from the Sparta detector so downstream agents can decide when to request alternates.

These files are referenced via `blob_path`/`rolling_windows_path` in `FetchResult.metadata`, giving downstream tools (extractor, MCPs, etc.) a deterministic attachment to hand to format-specific providers.

## Relationship to LiteLLM / SciLLM

The code lives here to make experimentation easier, but nothing prevents you from installing both `scillm` and `fetcher` in the same virtual environment. When the optional `alternates` extra is installed the package will automatically enable SciLLM + Brave resolution just like the router.

### Using it from other repos

When another project needs the shared fetcher, add a path dependency in its `pyproject.toml`:

```
[tool.uv.workspace]
members = ["fetcher", ...]

[project.optional-dependencies]
ingestion = ["fetcher @ file:///home/graham/workspace/experiments/fetcher"]
```

This ensures every repo consumes the single source of truth. See [`AGENTS.md`](AGENTS.md) for integration guidance.

## Next steps

1. Review [`QUICKSTART.md`](QUICKSTART.md) for local setup via `uv`.
2. Plain `uv run python -m fetcher.workflows.fetcher --help` executes standalone fetch jobs (single URL fetches or batch inventory runs).
3. Follow [`FEATURES.md`](FEATURES.md) for deep dives on policy knobs, paywall heuristics, and telemetry.
