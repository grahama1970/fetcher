# Fetcher: Consumer CLI (Surf-CLI–style) + ETL Entry Alias (Least Brittle)

Last updated: 2026-01-02  
Repo: https://github.com/grahama1970/fetcher  
UX inspiration (structure/discoverability): https://github.com/nicobailon/surf-cli

---

Status: Implemented in fetcher 0.1.13 (consumer CLI, forced downloads, summaries).
Implementation notes:
- Consumer runner uses shared modules; orchestration lives in `src/fetcher/consumer.py`.
- Forced downloads are implemented via `persist_downloads(... allow_junk=True)` (consumer-only exception).
- Final URL is recorded as `final_url` in metadata (aiohttp, Playwright, Wayback, Jina, Google Translate).
- Console scripts: `fetcher = fetcher.cli:app`, `fetcher-etl = fetcher.workflows.fetcher:main`.

## 0) Objective

Add a **purpose-built consumer CLI** that:
- is easy to use (“just work”, best-effort)
- has **fewer knobs**
- benefits from **all fallback logic** of ETL (alternates + Playwright + paywall heuristics)
- produces **minimal, chat-friendly outputs**
- is automation friendly (`--json`, stable schema)

Keep ETL behavior unchanged and accessible.

---

## 1) Strategy (sanest / least brittle)

### 1.1 Two entrypoints, one package
- `fetcher` = new consumer CLI (Typer)
- `fetcher-etl` = console script alias to existing ETL CLI entrypoint (**Option A**: unchanged behavior)

### 1.2 Two runners, shared modules (allowed divergence)
Consumer and ETL may **diverge at orchestration/runner level**, but MUST share the same core modules:
- network fetching (including Playwright integration)
- alternates/fallback resolution
- content evaluation (quality, paywall)
- artifact materialization helpers for text/markdown (no new output system)

**Rule:** do not duplicate fetch or extraction logic; call shared modules. Divergence is allowed in:
- what gets written as reports
- always-download behavior for consumer (even if junk)
- minimal summary formatting and UX

---

## 2) Non-negotiables (anti-bloat)

### 2.1 No LLM in v1
No Chutes/SciLLM or other LLM calls.

### 2.2 Fewer knobs by design
Consumer CLI exposes only:
- `--out <DIR>`
- `--emit <CSV>` (high-level toggles)
- `--json`
- `--soft-fail`
Plus discoverability:
- `--help-full`
- `--find <query>`

No extra knobs for alternates/playwright/etc. Consumer must be best-effort and “just work”.

### 2.3 Consumer always downloads
Consumer runner MUST persist downloaded bytes to disk even when content verdict is not OK (best-effort).
If persistence fails, record the error in summary, but do not silently skip.

---

## 3) Deliverables

### 3.1 Consumer CLI (Typer)
Commands:
- `fetcher get <url>`
- `fetcher get-manifest <path-or->`

### 3.2 Minimal outputs
Consumer runner MUST write:
- `<run-dir>/consumer_summary.json` (authoritative, stable schema)
- `<run-dir>/Walkthrough.md` (generated deterministically from consumer_summary.json; no extra logic)

Consumer runner MUST NOT generate ETL-only heavy reports by default.

### 3.3 Reuse existing materializers for text/markdown (no new layout)
Consumer runner MUST use existing helpers (same as ETL uses) for:
- extracted text materialization
- markdown / fit-markdown materialization

No `urls/<id>/...` convenience directory in v1.

### 3.4 ETL entrypoint alias
Add `fetcher-etl` as a console-script entrypoint that runs the existing ETL CLI unchanged.
(Implemented as `fetcher.workflows.fetcher:main`.)

---

## 4) UX requirements (Surf-like)

### 4.1 Help tiers
- `fetcher --help`: minimal overview + examples, only common flags
- `fetcher --help-full`: expanded usage, env vars summary (read-only), artifacts, troubleshooting, and clear pointer to `fetcher-etl`

### 4.2 `--find <query>` (discoverability)
Search:
- commands
- flags/options
- important env vars (existing, documented)
- artifact names and files produced

Output is one line per match:
`<category> <name> - <short description>`

---

## 5) Run directory semantics

### 5.1 `--out` semantics (explicit)
- If `--out <DIR>` is provided:
  - use exactly `<DIR>` as the run directory (do not create subdir)
- If omitted:
  - create `run/artifacts/<run-id>/`

### 5.2 Run id format
When `--out` omitted, generate UTC:
- `YYYYMMDDTHHMMSSZ_<6hex>`
Example: `20260102T153045Z_a1b2c3`

---

## 6) Consumer commands (exact behavior)

### 6.1 `fetcher get <url>`
Signature:
```
fetcher get <url> [--out <DIR>] [--emit <CSV>] [--json] [--soft-fail]
```

Behavior:
- single URL only
- run full best-effort fallback stack:
  - HTTP redirects
  - alternates resolvers (same shared modules/logic as ETL)
  - Playwright when heuristics trigger (same gating as ETL; best-effort if not installed)
- run content evaluation (quality/paywall verdicts) using shared modules (same as ETL)
- ALWAYS attempt to persist the downloaded file to disk
- materialize text/markdown outputs per `--emit`
- write `consumer_summary.json` and `Walkthrough.md`
- if `--json`, print ONLY the JSON object for that URL to stdout

### 6.2 `fetcher get-manifest <path-or->`
Signature:
```
fetcher get-manifest <urls.txt|-> [--out <DIR>] [--emit <CSV>] [--json] [--soft-fail]
```

Manifest format (strict v1):
- UTF-8
- one URL per non-empty, non-comment line
- blank lines ignored
- lines starting with `#` ignored
- NO inline metadata in v1 (URL only)

Stdin support:
- if `<path-or->` is `-`, read URLs from stdin

Behavior:
- run as one batch run (single run dir)
- same fallback stack + evaluation as `get`
- ALWAYS attempt to persist each downloaded file
- materialize text/markdown outputs per `--emit`
- write `consumer_summary.json` and `Walkthrough.md`
- if `--json`, print ONLY the batch JSON object to stdout

---

## 7) `--emit` semantics (high-level toggles, few knobs)

Allowed values:
- `download` (default ON): persist downloaded bytes
- `text`: materialize extracted text files
- `md`: materialize markdown
- `fit_md`: materialize fit-markdown (only meaningful if md enabled)

Default:
- `download,text,md`

Rules:
- Consumer runner MUST NOT implement extraction/materialization itself.
- It MUST call existing materialization helpers and rely on `FetchResult.metadata` output paths.

---

## 8) Provenance fields (required)

Consumer summary MUST include, per item:
- `original_url` (user input)
- `requested_url` (URL passed into fetch stage after inventory normalization/alternates selection, if applicable)
- `final_downloaded_url` (actual fetched URL after redirects and alternates)

If `final_downloaded_url` is not currently recorded by shared modules, add it to result metadata at the network layer (post-response URL).
(Implemented as `final_url` in metadata and surfaced in summaries.)

---

## 9) JSON schemas (authoritative)

### 9.1 `consumer_summary.json` (batch or single)
Top-level:
```json
{
  "command": "get" ,
  "run_id": "20260102T153045Z_a1b2c3",
  "run_dir": "run/artifacts/20260102T153045Z_a1b2c3",
  "started_at": "2026-01-02T15:30:45Z",
  "finished_at": "2026-01-02T15:31:12Z",
  "duration_ms": 27000,
  "counts": {
    "total": 1,
    "downloaded": 1,
    "ok": 1,
    "failed": 0,
    "used_playwright": 0,
    "used_alternates": 0
  },
  "items": [
    {
      "original_url": "https://example.com/a",
      "requested_url": "https://example.com/a",
      "final_downloaded_url": "https://example.com/a?x=1",
      "status": 200,
      "content_type": "text/html",
      "method": "aiohttp|playwright|alternate",
      "alternate_provider": "wayback|brave|jina|null",
      "verdict": "ok|junk|failed",
      "paywall_verdict": "ok|paywalled|unknown|null",
      "artifacts": {
        "download_path": "run/artifacts/.../downloads/<...>",
        "extracted_text_path": "run/artifacts/.../extracted_text/<...>.txt",
        "markdown_path": "run/artifacts/.../markdown/<...>.md",
        "fit_markdown_path": null
      },
      "warnings": [],
      "errors": []
    }
  ]
}
```

Notes:
- For `get`, `counts.total` is 1 and `items` length is 1.
- For `get-manifest`, `command` is `get-manifest` and `counts.total` is N.
- `method` and `alternate_provider` should be best-effort derived from existing metadata; if unknown, set null.

### 9.2 `--json` stdout output
- For `get --json`: print exactly the single run `consumer_summary.json` object (still with `items` length 1).
- For `get-manifest --json`: print exactly the batch run object.

No extra logging to stdout in `--json` mode.

---

## 10) Walkthrough.md (deterministic rendering)

Write `<run-dir>/Walkthrough.md` generated only from `consumer_summary.json`:
- header with run id, timestamps, duration
- counts table
- per-item section with:
  - original/requested/final url
  - status/verdict/paywall
  - artifact paths
  - errors/warnings (if any)

No additional inference or transformation beyond formatting.

---

## 11) Exit codes

- 0:
  - `get`: success if the run completed and at least `consumer_summary.json` exists
  - `get-manifest`: success if run completed and `consumer_summary.json` exists
  - with `--soft-fail`, allow per-item failures without non-zero exit
- 2: CLI usage error
- 3: fatal runtime error (cannot create run dir, crash)

---

## 12) Packaging changes (pyproject.toml)

- Add Typer dependency.
- Add console scripts:
  - `fetcher = "fetcher.cli:app"`
  - `fetcher-etl = "fetcher.workflows.fetcher:main"`

---

## 13) Implementation checklist (agent-ready)

1. Identify existing ETL CLI entrypoint and wire `fetcher-etl` to it (no behavior changes).
2. Implement `src/fetcher/cli.py` (Typer) with:
   - `get`
   - `get_manifest` (file path or stdin via `-`)
   - `--help-full`
   - `--find`
3. Implement consumer runner orchestration:
   - uses shared fetch + alternates + Playwright gating modules
   - runs content evaluation
   - always persists download bytes
   - calls existing materialization helpers for text/markdown
4. Write `consumer_summary.json` (stable schema) and `Walkthrough.md`.
5. Add tests for:
   - manifest parsing rules (comments/blank lines/stdin)
   - run-id format
   - JSON schema shape
   - Walkthrough generation is deterministic from JSON
