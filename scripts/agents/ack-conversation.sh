#!/usr/bin/env bash
set -euo pipefail

# Ack a conversation message after successful handling.
# Required: MSG_ID (e.g., agent_conversations/abcdef123456)

MSG_ID=${MSG_ID:?"MSG_ID is required (agent_conversations/<key>)"}
AGENT=${AGENT:-fetcher}

ARANGO_URL="${ARANGO_URL:-http://127.0.0.1:8529}"
ARANGO_DB="${ARANGO_DB:-lessons}"
ARANGO_USER="${ARANGO_USER:-root}"
ARANGO_PASS="${ARANGO_PASS:-openSesame}"
MEMORY_SRC="${MEMORY_SRC:-/home/graham/workspace/experiments/memory/src}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHONPATH="$MEMORY_SRC:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH ARANGO_URL ARANGO_DB ARANGO_USER ARANGO_PASS

python -m typer graph_memory.lessons.agent_conversations run ack \
  --id "$MSG_ID" \
  --agent "$AGENT" \
  --json
