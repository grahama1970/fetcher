# D3FEND category URL hang + thin extraction (Fetcher)

## Summary

Observed symptoms when fetching D3FEND category/technique pages such as:
- `https://d3fend.mitre.org/tactic/d3f:Evict/`
- `https://d3fend.mitre.org/technique/d3f:FileEviction/`

1. **Fetcher appears to “hang” / times out** when callers expect a quick single-URL result.
2. In environments where Playwright is not available, **HTML fetches can return very little visible text**, leading to extremely short extracted content.

## Root Cause(s)

### 1) Link-hub fan-out triggering on non-hub pages

`URLFetcher.fetch_many()` includes a Phase 2 “hub fan-out” that can enqueue and fetch many child links after the initial page fetch.

The previous hub heuristic treated **any page with >= `SPARTA_TOC_LINKS_MIN` links** (default 20) as a hub, even when the page had substantial narrative text.

On D3FEND pages this can produce large `link_hub_links` lists, and Phase 2 fetching can dominate runtime, which looks like a “hang” to higher-level tools with short timeouts.

### 2) JS-heavy pages when Playwright is unavailable

If a page needs JS rendering and Playwright is not installed/available, a direct HTTP fetch may be content-thin. A pure readability-style extraction can then produce <100 chars.

## Fix Implemented (in this repo)

- Refined hub classification so pages are only marked `link_hub=true` when they are meaningfully link-heavy relative to body text (low body text **or** high link-density), not merely “many links”.
- Added `FetchConfig.enable_toc_fanout` (optional) to explicitly toggle Phase 2 fan-out without relying on global env.
- `fetcher.workflows.fetcher.fetch_url(...)` now disables fan-out by default (single-URL calls should not unexpectedly fetch dozens of children).
- Added CLI flag `--no-fanout` to disable fan-out per run.
- When a page is JS-heavy but Playwright is unavailable, fetcher now attempts a fallback to `r.jina.ai` to get usable text.

## Quick Repro

### Before fix

Single-URL helpers could take tens of seconds because Phase 2 fan-out ran and fetched many children.

### After fix

Run:

```bash
uv run python -m fetcher.workflows.fetcher --url https://d3fend.mitre.org/tactic/d3f:Evict/
uv run python -m fetcher.workflows.fetcher --url https://d3fend.mitre.org/technique/d3f:FileEviction/
```

Optional: explicitly disable fan-out:

```bash
uv run python -m fetcher.workflows.fetcher --no-fanout --url https://d3fend.mitre.org/tactic/d3f:Evict/
```

## Notes for Downstream Pipelines (Step 06b)

Short-term workaround (“pass URL directly to Kimi”) should no longer be necessary for these D3FEND pages when using `fetch_url(...)` or the CLI with `--no-fanout` as needed.
