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

Need to refresh chronic mirrors (NASA links, DTIC PDFs, etc.)? Run the maintained CLI:

```
uv run python -m fetcher.tools.mirror_refresher \
  --manifest src/fetcher/data/mirror_sources.json \
  --out src/fetcher/data/local_sources
```

The manifest lists brittle URLs; the refresher fetches and writes them under `local_sources/` so overrides can reference file:// URIs deterministically.

By default `FETCHER_TEXT_INLINE_MAX_BYTES=0`, so **all** response bodies are externalized to
`run/artifacts/text_blobs/` and the JSON metadata only stores a pointer via `file_path`
(`text_path` remains for backward compatibility). Set the environment variable to a positive byte
threshold (e.g., `FETCHER_TEXT_INLINE_MAX_BYTES=200000`) if you want small payloads to remain
inline inside the `.results.jsonl` file. When text is externalized, fetcher records
`text_externalized`, `text_inline_missing`, and `text_length_chars` on each result so downstream
pipelines know the original length even though the inline `text` field is blank. If you want a
shared blob cache across runs, set `FETCHER_TEXT_CACHE_DIR=/path/to/cache` and fetcher will dedupe
by SHA-256 (run artifacts link to the cached file).

Every `FetchResult` now also records `content_sha256`, `content_previous_sha256`,
`content_changed`, and a `content_diff_ratio` when a prior snapshot exists. Hashes are tracked in
`run_artifacts/fetch_cache/content_hashes.json`, making it trivial to see whether DOM/text changed
between runs.

## Download modes & rolling windows

Use `FETCHER_DOWNLOAD_MODE` to control how response bodies are stored:

- `text` (default): records only metadata in JSON and stores the raw body under `file_path` unless you
  explicitly raise `FETCHER_TEXT_INLINE_MAX_BYTES`.
- `download_only`: persist every body (binary or text) to `run/artifacts/downloads/<sha>.<ext>` and zero the inline `text` field.
- `rolling_extract`: keep the download + emit JSONL rolling windows under `run/artifacts/rolling_windows/` using `FETCHER_ROLLING_WINDOW_SIZE`, `FETCHER_ROLLING_WINDOW_STEP`, and optional `FETCHER_ROLLING_WINDOW_MAX_WINDOWS`. Windows follow spaCy sentence boundaries when spaCy is installed (otherwise fall back to a regex splitter) so no chunk cuts off a sentence mid-way.

Each HTML result also records a `paywall_detection` blob (score, verdict, indicators) derived from the Sparta detector so downstream agents can decide when to request alternates.

These files are referenced via `blob_path`/`rolling_windows_path` in `FetchResult.metadata`, giving downstream tools (extractor, MCPs, etc.) a deterministic attachment to hand to format-specific providers.

### HTTP cache defaults

Fetcher now enables the HTTP cache by default so repeated requests for the same URL (with identical normalized paths) reuse the last good snapshot instead of re-downloading HTML every time. Controls:

- `FETCHER_HTTP_CACHE_DIR` (default `run/fetch_cache`) or `FETCHER_HTTP_CACHE_PATH` (explicit file) pick the on-disk cache location.
- `FETCHER_HTTP_CACHE_DISABLE=1` or CLI `--no-http-cache` turns the feature off for runs that must ignore cached snapshots.
- `FETCHER_HTTP_CACHE_PATH` can be shared across processes to seed downstream agents; cached rows are skipped automatically when they were produced by deprecated shortcuts (e.g., the removed D3FEND CSV mirror).

Single-URL helpers (`fetcher.workflows.fetcher.fetch_url`) also respect the same environment variables, and every `FetchResult` advertises whether it came from cache via the `from_cache` field.

### Optional PDF discovery

Many reference pages (MITRE D3FEND tactics, ATT&CK summaries, etc.) embed “Download PDF” links that contain richer content than the HTML summary. Set `FETCHER_ENABLE_PDF_DISCOVERY=1` (or pass `--pdf-discovery N` to the CLI) to have fetcher automatically queue up to `FETCHER_PDF_DISCOVERY_MAX` (default 3) PDF links discovered on each HTML page. Auto-fetched PDFs inherit the parent metadata (`parent_url`, `source="pdf_discovery"`, `pdf_discovered=true`), respect the same throttling/proxy rules, and appear alongside the original page in the run artifacts. Leave the flag off to fetch only the URLs you explicitly requested.

### Optional PDF password cracking

If you need to unlock lightly protected PDFs (short PINs/dictionaries), install the optional helper and enable the env toggle:

```
uv add pdferli  # or pip install pdferli
export FETCHER_PDF_CRACK_ENABLE=1
export FETCHER_PDF_CRACK_CHARSET=0123456789
export FETCHER_PDF_CRACK_MINLEN=4
export FETCHER_PDF_CRACK_MAXLEN=6
# optional: FETCHER_PDF_CRACK_PROCESSES, FETCHER_PDF_CRACK_TIMEOUT, FETCHER_PDF_BRUTE_LIMIT
```

Fetcher will attempt a bounded brute-force when `pdf.needs_pass` is detected. Successful cracks proceed with normal extraction; failures are tagged `content_verdict="password_protected"` and skip downstream chunking. When `pdferli` cannot recover a password but the search space is small, `FETCHER_PDF_BRUTE_LIMIT` controls an optional PyMuPDF fallback (default 50 000 combinations).

### Progress logging

Long fetch batches now emit `[fetcher] progress <completed>/<total>: …` lines to stderr so you can confirm the run is alive even when piping through `tee`. Tune the cadence via `FETCHER_PROGRESS_INTERVAL` (default `50`) or disable the logs with `FETCHER_PROGRESS_ENABLE=0`.

```
FETCHER_PROGRESS_INTERVAL=25 make fetch_sparta_urls
```

## Link-hub fan-out

When a page is flagged as a link hub (ATT&CK mitigations, NIST glossary tables, etc.) fetcher can
automatically queue the linked resources (one level deep). Controls:

- `SPARTA_TOC_FANOUT=1` (default) enables the behavior.
- `SPARTA_TOC_PATH_ALLOWLIST_JSON` restricts fan-out to explicit `{ "host": ["/prefix"] }` paths.
- `SPARTA_TOC_ALLOWLIST_DISABLE_DEFAULTS=1` disables the built-in CSRC path list.
- `SPARTA_TOC_FANOUT_DOMAINS` (comma-separated suffixes) lets entire domains fan out regardless of
  path. Defaults include `mitre.org`, `d3fend.mitre.org`, `sparta.aerospace.org`, and `nist.gov`, so
  ATT&CK/D3FEND/Sparta/NIST hubs automatically fetch their same-domain children.

## Rotating proxy fallback (Step 06)

Step 06 crawls can now re-issue rate-limited requests through a rotating proxy (e.g., IPRoyal) before falling back to Wayback. The loader understands the same env variables documented in `memory/scripts/smokes/iproyal_login.py` and `graph-memory-operator/scripts/smokes/iproyal_curl_smoke.sh`, so you can reuse existing secrets.

```
export SPARTA_STEP06_PROXY_HOST=gw.iproyal.com
export SPARTA_STEP06_PROXY_PORT=12321
export SPARTA_STEP06_PROXY_USER=team
export SPARTA_STEP06_PROXY_PASSWORD=super-secret
# Optional knobs
export SPARTA_STEP06_PROXY_DOMAINS=d3fend.mitre.org,atlas.mitre.org
export SPARTA_STEP06_PROXY_STATUSES=429,403
export SPARTA_STEP06_PROXY_HINTS="rate limit,too many requests"
```

- `SPARTA_STEP06_PROXY_URL=http://user:pass@gw.iproyal.com:12321` or `SPARTA_STEP06_PROXY_CREDENTIALS=user:pass` can replace the separate host/user/pass envs.
- Defaults only target `d3fend.mitre.org`; override with `SPARTA_STEP06_PROXY_DOMAINS` or disable the default via `SPARTA_STEP06_PROXY_DOMAINS_DISABLE_DEFAULTS=1`.
- Trigger statuses default to `429`; textual hints catch throttling pages ("rate limit", "retry later").
- Set `SPARTA_STEP06_PROXY_DISABLE=1` to turn the feature off without clearing secrets. If Step 06 envs are missing the loader falls back to `IPROYAL_HOST|PORT|USER|PASSWORD` so the smoke scripts and fetcher stay aligned.

Whenever a throttled response matches the allowlist, Fetcher retries the same URL through the proxy. Proxied rows add `proxy_rotation_*` metadata to each `FetchResult`, and the Step 06 audit JSON inherits a `proxy_rotation` summary (`attempted`, `success`, per-domain breakdown, recent errors) so operators can trace every rotation alongside the existing TOC telemetry.

> Repo default: `.env` ships with an IPRoyal pool from the `memory` project (`SPARTA_STEP06_PROXY_*` + `IPROYAL_*` entries pointing at `geo.iproyal.com:12321`). Update these values locally if your tenant rotates credentials, and keep `IPROYAL_PROXY_LIST=iproyal-proxies.txt` in sync.

## Run artifacts

Every batch run writes telemetry to `run/artifacts/<run-id>/`, including:

- `outstanding_controls_remaining.json`, `outstanding_urls_summary.json`, `outstanding_domains_summary.json`
- `alternate_urls_applied.jsonl` (when Brave/Wayback alternates succeed)
- `junk_results.jsonl` + `junk_summary.json` (every non-`ok` `content_verdict` for human/agent review)
- `junk_table.md` (bounded markdown table for quick triage of junk URLs)
- `text_blobs/` and `downloads/` (the actual files referenced by `file_path` in the `.results.jsonl` output)
- `extracted_text/` (clean text artifacts; each `FetchResult` points to `extracted_text_path` when enabled)
- `markdown/` (LLM-friendly markdown artifacts; enable via `FETCHER_EMIT_MARKDOWN=1`, stored under `markdown_path`)
- `fit_markdown/` (pruned markdown artifacts for LLM input; emitted only when markdown is enabled and `FETCHER_EMIT_FIT_MARKDOWN=1`, stored under `fit_markdown_path`)
- `changes.jsonl` – one line per URL whose content changed this run, including previous hash/timestamp and the latest diff ratio.

For one-off debugging of “raw HTML vs markdown vs what the browser rendered”, use:
`uv run python scripts/compare_markdown_to_html.py --url <URL> --run-dir run/artifacts/<name>`

## Consumer CLI (best-effort)

The consumer CLI is a simplified, surf-style interface for quick, human-friendly runs.
It always tries to persist downloads (even when content is junk) and emits only a
minimal summary + walkthrough by default.

Commands:

```
fetcher get <url> [--out <DIR>] [--emit <CSV>] [--json] [--soft-fail]
fetcher get-manifest <urls.txt|-> [--out <DIR>] [--emit <CSV>] [--json] [--soft-fail]
```

Key behaviors:
- Always attempts to persist downloads to `downloads/` in the run directory.
- Uses the same fetch + alternates + Playwright logic as ETL, but with fewer knobs.
- Writes `<run-dir>/consumer_summary.json` and `<run-dir>/Walkthrough.md`.
- `--json` prints only the summary JSON to stdout (no extra logs).
- `--dry-run` validates inputs + environment without fetching.
- `--help-full` and `--find <query>` provide discoverability.
- Missing Brave/Playwright dependencies surface as environment warnings (stderr + summary/audit).
- `--soft-fail` keeps exit code 0 even when environment warnings are present; otherwise warnings raise a non-zero exit.
ETL runs (`fetcher-etl`) accept `--soft-fail` to keep exit code 0 when environment warnings are present.

`fetcher-etl` remains the full ETL CLI with all knobs and reports.

### ETL metrics (opt-in)
Use these flags with `fetcher-etl` to emit lightweight reliability metrics:

```
fetcher-etl --inventory urls.jsonl --metrics-json metrics.json
fetcher-etl --inventory urls.jsonl --metrics-json - --print-metrics
```

`--metrics-json -` writes the metrics object to stdout (suppressing the default summary).

Audit files now also embed `rate_limit_metrics` (overall runtime, effective RPS, per-domain 429 stats)
and, when configured, a `proxy_rotation` block so operators can trace throttling and rotation usage
without spelunking logs.

### Diagnostics & smokes
- `fetcher doctor` (or `fetcher-etl --doctor`) prints environment/dependency checks with redacted values.
- `fetcher get --dry-run <url>` or `fetcher-etl --dry-run --inventory <path>` validate inputs without fetching.
- `fetcher-etl --help`, `fetcher-etl --help-full`, and `fetcher-etl --find <query>` mirror the surf-style discoverability of `fetcher`.
- A small smoke inventory lives at `docs/smokes/urls.jsonl` for quick ETL checks.

## Relationship to LiteLLM / SciLLM

The code lives here to make experimentation easier, but nothing prevents you from installing both `scillm` and `fetcher` in the same virtual environment. When the optional `alternates` extra is installed the package will automatically enable SciLLM + Brave resolution just like the router. The `alternates` extra is pinned to a specific `scillm` GitHub SHA for reproducibility; update the pin in `pyproject.toml` if you want newer SciLLM behavior (the default branch is `main`).

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
