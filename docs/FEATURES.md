# Fetcher Feature Set

## Core fetch loop
- Async HTTP via `aiohttp` with per-domain throttles (`concurrency`, `per_domain`, `max_attempts`).
- Cache-first text extraction with SHA tracking to dedupe repeated runs.
- Inline Playwright fallback for SPA pages plus optional OCR for PDF/image heavy sites.

## Prefilters & validation
- HTML/paragraph thresholds keep bad bodies from reaching downstream chunking.
- Configurable heading-density gate catches listicles.
- Central `FetcherPolicy` exposes overrides for paywall domains, IDNA normalization, and per-domain strip rules.

## Overrides & alternates
- `data/overrides.json` ships deterministic replacements for brittle vendors (OWASP, MITRE ATLAS, NASA, etc.).
- Local mirror resolution, path regex routing, and Brave search suggestions share a single audit log.
- Optional SciLLM JSON-mode workflow (`CHUTES_*` env) generates contextual alternates tied to control metadata. The `fetcher[alternates]` extra is pinned to a specific `scillm` GitHub SHA for reproducibility; update the pin in `pyproject.toml` if you want newer SciLLM behavior (the default branch is `main`).

## Rate-limit mitigation
- Step 06 proxy rotation: when `SPARTA_STEP06_PROXY_*` env vars (or `IPROYAL_*` fallbacks) are present, throttled requests to allowlisted domains (default `d3fend.mitre.org`) are automatically retried through the configured rotating proxy before Wayback/Jina.
- Triggers are configurable via `SPARTA_STEP06_PROXY_DOMAINS`, `SPARTA_STEP06_PROXY_STATUSES`, and `SPARTA_STEP06_PROXY_HINTS`; disable defaults with `SPARTA_STEP06_PROXY_DOMAINS_DISABLE_DEFAULTS=1` or turn the feature off entirely via `SPARTA_STEP06_PROXY_DISABLE=1`.
- Fetch results carry `proxy_rotation_*` metadata (provider, reason, endpoint, error) and the Step 06 audit gains a `proxy_rotation` summary (attempts, successes, per-domain counts) so Ops can trace exactly when IP rotation was used.
- Audits also include `rate_limit_metrics` (runtime + effective RPS + per-domain 429 counts) so you can spot slow domains before rotation is required.

## Content tracking
- Every `FetchResult` records `content_sha256`, `content_previous_sha256`, `content_changed`, and a `content_diff_ratio` (when prior samples exist). The shared cache lives under `run_artifacts/fetch_cache/content_hashes.json`.
- Optional shared blob cache: set `FETCHER_TEXT_CACHE_DIR=/path/to/cache` and text blobs will dedupe across runs via SHA-256. Metadata points to the cached path so downstream agents can rehydrate bodies without copying them per run.

## Mirror refresher
- Run `uv run python -m fetcher.tools.mirror_refresher` to hydrate `src/fetcher/data/local_sources/` from the declarative manifest (`src/fetcher/data/mirror_sources.json`). This keeps override targets in sync for brittle NASA/DTIC/D3FEND sources.
- Each manifest entry specifies `url` + `path`; the refresher downloads the byte stream, verifies SHA-256, and writes to the relative path so overrides can reference `file://` URIs.

## Outputs & auditability
- `FetchResult` persists text digests and metadata to JSONL, suitable for knowledge stores.
- `FETCHER_DOWNLOAD_MODE` controls whether bodies stay inline (`text`), are mirrored to disk (`download_only`), or mirrored plus chunked into rolling windows for stream ingestion (`rolling_extract`). Rolling windows respect spaCy sentence boundaries when available (regex fallback otherwise).
- `FetcherResult.audit` summarizes outstanding controls, applied alternates, and per-domain failures.
- Structured keys live in `fetcher.core.keys` so integrations never stringify magic keys.
- `outstanding_domains_summary.json` groups unresolved URLs by category, including `paywall`, `content_thin`, `link_hub`, `bot_blocked` (anti-bot interstitials), `password_protected`, `missing_file`, and `needs_login_or_playwright`.
- Encrypted/password-protected PDFs: when `FETCHER_PDF_CRACK_ENABLE=1` and the optional `pdferli` dependency is installed, fetcher will attempt a bounded brute-force crack (charset/min/max length configurable via env) before extraction; otherwise results are marked `content_verdict="password_protected"` with empty text so downstream chunking/LLM fan-out can deterministically skip them.

## Diagnostics
- `fetcher doctor` (or `fetcher-etl --doctor`) prints environment/dependency checks with redacted values.
- `--dry-run` validates inputs and policy without fetching.
- `fetcher-etl --help-full` and `--find <query>` provide surf-style discoverability for the ETL CLI.
- `docs/smokes/urls.jsonl` ships a small smoke inventory for quick ETL checks.

## Extensibility hooks
- Policy injection allows custom paywall hints, alternate providers, or validator tuning.
- Env vars (`SPARTA_*`, `FETCHER_*`) let Ops teams tweak thresholds without redeploying code.
- Router-safe: embed inside LiteLLM proxy, MCP servers, or stand-alone ETL jobs.
