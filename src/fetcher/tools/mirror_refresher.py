"""Mirror refresher CLI for deterministic local copies of brittle sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import requests

logger = logging.getLogger(__name__)

ManifestEntry = Dict[str, str]
FetchFunc = Callable[[str], bytes]


def _default_fetch(url: str, *, timeout: int = 30) -> bytes:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.content


def load_manifest(path: Path) -> List[ManifestEntry]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Iterable):
        raise ValueError("mirror manifest must be a list")
    entries: List[ManifestEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not item.get("url"):
            continue
        entries.append(item)
    return entries


def refresh_manifest(
    manifest_path: Path,
    output_dir: Path,
    *,
    fetch: Optional[FetchFunc] = None,
    dry_run: bool = False,
    timeout: int = 30,
) -> List[Dict[str, str]]:
    fetcher = fetch or (lambda url: _default_fetch(url, timeout=timeout))
    entries = load_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, str]] = []

    for entry in entries:
        url = entry.get("url", "").strip()
        rel = (entry.get("path") or entry.get("filename") or "").strip()
        if not url or not rel:
            continue
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            logger.info("dry-run: would refresh %s -> %s", url, target)
            results.append({"url": url, "path": str(target), "status": "skipped"})
            continue
        payload = fetcher(url)
        sha = hashlib.sha256(payload).hexdigest()
        target.write_bytes(payload)
        logger.info("refreshed %s (%d bytes) -> %s", url, len(payload), target)
        results.append({
            "url": url,
            "path": str(target),
            "bytes": str(len(payload)),
            "sha256": sha,
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh mirrored local sources for fetcher overrides")
    parser.add_argument(
        "--manifest",
        default=Path("src/fetcher/data/mirror_sources.json"),
        type=Path,
        help="Path to mirror manifest JSON (default: src/fetcher/data/mirror_sources.json)",
    )
    parser.add_argument(
        "--out",
        default=Path("src/fetcher/data/local_sources"),
        type=Path,
        help="Directory to write mirrored artifacts (default: src/fetcher/data/local_sources)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout per URL (seconds)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    refresh_manifest(args.manifest, args.out, dry_run=args.dry_run, timeout=args.timeout)


if __name__ == "__main__":
    main()
