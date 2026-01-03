#!/usr/bin/env bash
set -euo pipefail

uv venv --python 3.12 /tmp/fetcher-alternates-smoke
source /tmp/fetcher-alternates-smoke/bin/activate
uv pip install ".[alternates]"
python -c "import scillm; print('scillm import ok')"
deactivate
rm -rf /tmp/fetcher-alternates-smoke
echo "[smoke] alternates install passed"
