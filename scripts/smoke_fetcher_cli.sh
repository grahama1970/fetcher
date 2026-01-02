#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

KEEP="${KEEP:-0}"
TMP_DIR="$(mktemp -d -t fetcher-smoke-XXXXXX)"
cleanup() {
  if [ "$KEEP" != "1" ]; then
    rm -rf "$TMP_DIR"
  else
    echo "[smoke] keeping artifacts at $TMP_DIR"
  fi
}
trap cleanup EXIT

echo "[smoke] temp dir: $TMP_DIR"

echo "[smoke] fetcher --help"
uv run fetcher --help >/dev/null

echo "[smoke] fetcher --help-full"
uv run fetcher --help-full >/dev/null

echo "[smoke] fetcher --find markdown"
uv run fetcher --find markdown >/dev/null

echo "[smoke] fetcher get (single url)"
RUN_DIR="$TMP_DIR/consumer_single"
uv run fetcher get https://www.nasa.gov --out "$RUN_DIR" >/dev/null
test -f "$RUN_DIR/consumer_summary.json"
test -f "$RUN_DIR/Walkthrough.md"
test -d "$RUN_DIR/downloads"

echo "[smoke] fetcher get-manifest (file)"
MANIFEST="$TMP_DIR/urls.txt"
printf "https://www.nasa.gov\nhttps://example.com\n" > "$MANIFEST"
RUN_DIR="$TMP_DIR/consumer_manifest"
uv run fetcher get-manifest "$MANIFEST" --out "$RUN_DIR" >/dev/null
test -f "$RUN_DIR/consumer_summary.json"
test -f "$RUN_DIR/Walkthrough.md"
test -d "$RUN_DIR/downloads"

echo "[smoke] fetcher get-manifest (stdin)"
RUN_DIR="$TMP_DIR/consumer_manifest_stdin"
cat "$MANIFEST" | uv run fetcher get-manifest - --out "$RUN_DIR" >/dev/null
test -f "$RUN_DIR/consumer_summary.json"
test -f "$RUN_DIR/Walkthrough.md"
test -d "$RUN_DIR/downloads"

echo "[smoke] fetcher get --json (stdout only)"
JSON_OUT="$TMP_DIR/consumer_json.json"
uv run fetcher get https://example.com --out "$TMP_DIR/consumer_json" --json > "$JSON_OUT"
python - "$JSON_OUT" <<'PY'
import json, sys
path = sys.argv[1] if len(sys.argv) > 1 else None
if not path:
    raise SystemExit("missing json path")
with open(path, "r", encoding="utf-8") as fh:
    json.load(fh)
print("[smoke] json ok:", path)
PY

echo "[smoke] fetcher-etl --help"
uv run fetcher-etl --help >/dev/null

echo "[smoke] fetcher-etl --url (single url)"
ETL_OUT="$TMP_DIR/etl_smoke.results.jsonl"
uv run fetcher-etl --url https://example.com --output "$ETL_OUT" --run-artifacts "$TMP_DIR/etl_artifacts" >/dev/null
test -f "$ETL_OUT"

echo "[smoke] done"
