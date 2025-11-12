# AGENT INSTRUCTIONS — FETCHER

Fetcher is the shared web-ingestion toolkit used across Sparta, SciLLM, and LiteLLM agents. When editing or calling this
module, keep the following in mind:

## PURPOSE
- Provide deterministic HTML/PDF retrieval, including Brave fallbacks, paywall heuristics, and override mirrors.
- Normalize outputs (domain, titles, control metadata) so downstream automation can diff results across repositories.
- Remain provider-agnostic; no SciLLM runtime dependencies unless callers install the optional `alternates` extra.

## EXPECTATIONS FOR AGENTS
1. **Respect policy knobs**: Runtime behavior is configured via `fetcher.workflows.fetcher.FetcherPolicy`. Never hardcode
   domain handling—extend overrides or the policy object instead.
2. **Keep data assets small**: Large mirrored artifacts belong in object storage. Store only deterministic fixtures inside
   `src/fetcher/data` (text/PDF <= 1 MB) so the package remains lightweight.
3. **Prefer composition**: Other repos should depend on this project via `path = "file:///home/graham/workspace/experiments/fetcher"`
   in their `pyproject.toml`. Avoid copying modules.
4. **Document new transports**: If you add a resolver (Wayback, Perplexity, etc.) update `FEATURES.md` and `QUICKSTART.md`
   with setup steps and environment variables.
5. **Test via uv**: Run `uv run pytest` or targeted scripts. The `.venv` in this repo is Python 3.12.10; match that when
   running local experiments from other projects.

## OPERATIONS
- Required env defaults live in `.env` (safe for local dev, not production secrets).
- `uv run python -m fetcher.workflows.fetcher --help` exposes the CLI entrypoint when you need to fetch a URL manually.
- When integrating with agents, call `fetcher.workflows.fetcher.fetch_url(...)` instead of shelling out to
  `curl`.

## CHANGE MANAGEMENT
- Keep changes backwards compatible; several repos consume this via file path dependencies.
- Use semantic version bumps in `pyproject.toml` whenever you change the public API or default policy.
- For breaking schema updates, coordinate with Sparta/LiteLLM maintainers and document the migration steps.
