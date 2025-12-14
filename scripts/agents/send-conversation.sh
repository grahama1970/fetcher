#!/usr/bin/env bash
set -euo pipefail

# Send a conversation message.
# Required: ID_TO, BODY. Optional: TOPIC, PRIORITY (low|normal|high), ACTION_REQUIRED (true/false).

ID_TO=${ID_TO:?"ID_TO is required"}
BODY=${BODY:?"BODY is required"}
TOPIC=${TOPIC:-handoff}
PRIORITY=${PRIORITY:-normal}
ACTION_REQUIRED=${ACTION_REQUIRED:-false}
ID_FROM=${ID_FROM:-fetcher}

ARANGO_URL="${ARANGO_URL:-http://127.0.0.1:8529}"
ARANGO_DB="${ARANGO_DB:-lessons}"
ARANGO_USER="${ARANGO_USER:-root}"
ARANGO_PASS="${ARANGO_PASS:-openSesame}"
MEMORY_SRC="${MEMORY_SRC:-/home/graham/workspace/experiments/memory/src}"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PYTHONPATH="$MEMORY_SRC:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH ARANGO_URL ARANGO_DB ARANGO_USER ARANGO_PASS

python -m typer graph_memory.lessons.agent_conversations run add \
  --id-from "$ID_FROM" \
  --id-to "$ID_TO" \
  --topic "$TOPIC" \
  --priority "$PRIORITY" \
  $( [ "$ACTION_REQUIRED" = "true" ] && echo "--action-required" ) \
  --body "$BODY" \
  --json
