# Fetcher Data Assets

This package ships a curated `overrides.json` plus placeholder directories for optional runtime data:

- `local_sources/` – drop static mirrors or PDFs that should be returned instead of remote URLs. Files are looked up by exact filename.
- `processed/controls_context.jsonl` – optional knowledge context used when auto-generating alternate URLs. Each line should be a JSON object with at least an `id` field.

The directories are intentionally empty so downstream environments can mount their own data without having to modify package code.
