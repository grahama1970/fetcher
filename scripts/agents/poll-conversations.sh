#!/usr/bin/env bash
set -euo pipefail

# Poll for action-required messages for this agent (default: fetcher).
# Respects environment overrides; falls back to sensible defaults.

AGENT_NAME="${AGENT_NAME:-fetcher}"
ARANGO_URL="${ARANGO_URL:-http://127.0.0.1:8529}"
ARANGO_DB="${ARANGO_DB:-lessons}"
ARANGO_USER="${ARANGO_USER:-root}"
ARANGO_PASS="${ARANGO_PASS:-openSesame}"

# Allow override; fallback to local memory repo if installed in editable mode.
MEMORY_SRC="${MEMORY_SRC:-/home/graham/workspace/experiments/memory/src}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHONPATH="$MEMORY_SRC:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH ARANGO_URL ARANGO_DB ARANGO_USER ARANGO_PASS

python -m typer graph_memory.lessons.agent_conversations run list \
  --id-to "$AGENT_NAME" \
  --action-required \
  --priority "" \
  --limit 50 \
  --offset 0 \
  --json
