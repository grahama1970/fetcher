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
- Optional SciLLM JSON-mode workflow (`CHUTES_*` env) generates contextual alternates tied to control metadata.

## Outputs & auditability
- `FetchResult` persists text digests and metadata to JSONL, suitable for knowledge stores.
- `FETCHER_DOWNLOAD_MODE` controls whether bodies stay inline (`text`), are mirrored to disk (`download_only`), or mirrored plus chunked into rolling windows for stream ingestion (`rolling_extract`). Rolling windows respect spaCy sentence boundaries when available (regex fallback otherwise).
- `FetcherResult.audit` summarizes outstanding controls, applied alternates, and per-domain failures.
- Structured keys live in `fetcher.core.keys` so integrations never stringify magic keys.

## Extensibility hooks
- Policy injection allows custom paywall hints, alternate providers, or validator tuning.
- Env vars (`SPARTA_*`, `FETCHER_*`) let Ops teams tweak thresholds without redeploying code.
- Router-safe: embed inside LiteLLM proxy, MCP servers, or stand-alone ETL jobs.
