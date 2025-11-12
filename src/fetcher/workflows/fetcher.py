"""Reusable helpers for Sparta URL fetching and alternate resolution."""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import mimetypes
import os
import re
import sys
from dataclasses import dataclass
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set
from urllib.parse import urlparse

import requests

from .web_fetch import FetchConfig, FetchResult, URLFetcher, write_results
from .paywall_detector import detect_paywall
from .paywall_utils import resolve_paywalled_entries, reload_overrides_cache

try:  # Optional spaCy sentencizer
    import spacy  # type: ignore
except Exception:  # pragma: no cover - spaCy optional
    spacy = None  # type: ignore
from .fetcher_config import (
    BRAVE_ENDPOINT,
    HDR_ACCEPT,
    HDR_BRAVE_TOKEN,
    PAYWALL_DOMAINS,
    PAYWALL_STATUS_CODES,
    PAYWALL_HINTS,
    BRAVE_EXCLUDED_DOMAINS,
    LOCAL_SOURCES_DIR,
    OVERRIDES_PATH,
)
from ..core.keys import (
    K_URL,
    K_ORIGINAL_URL,
    K_DOMAIN,
    K_CONTROLS,
    K_TITLES,
    K_FRAMEWORKS,
    K_WORKSHEETS,
    K_TITLE,
    K_SOURCE,
    K_STATUS,
    K_TEXT_PATH,
)
from .fetcher_utils import (
    idna_normalize as _idna_normalize,
    normalize_domain as _normalize_domain,
    normalize_set as _normalize_set,
    resolve_repo_root,
)
from .download_utils import (
    annotate_paywall_metadata,
    apply_download_mode,
    maybe_externalize_text,
)


# ---------------------------------------------------------------------------
# Constants shared with the pipeline entrypoint.
# ---------------------------------------------------------------------------


BRAVE_ENDPOINT = BRAVE_ENDPOINT
HDR_ACCEPT = HDR_ACCEPT
HDR_BRAVE_TOKEN = HDR_BRAVE_TOKEN

PAYWALL_DOMAINS = PAYWALL_DOMAINS
PAYWALL_STATUS_CODES = PAYWALL_STATUS_CODES
PAYWALL_HINTS = PAYWALL_HINTS
BRAVE_EXCLUDED_DOMAINS = BRAVE_EXCLUDED_DOMAINS
LOCAL_SOURCES_DIR = LOCAL_SOURCES_DIR
OVERRIDES_PATH = OVERRIDES_PATH


def _env_int(name: str, default: int = 0) -> int:
    try:
        raw = os.getenv(name, "")
        return int(raw) if raw.strip() else default
    except (TypeError, ValueError):
        return default


TEXT_INLINE_MAX_BYTES = _env_int("FETCHER_TEXT_INLINE_MAX_BYTES", 0)


def _normalize_download_mode(value: str | None) -> str:
    mode = (value or "text").strip().lower()
    if mode not in {"text", "download_only", "rolling_extract"}:
        return "text"
    return mode


DOWNLOAD_MODE = _normalize_download_mode(os.getenv("FETCHER_DOWNLOAD_MODE"))
ROLLING_WINDOW_SIZE = max(1, _env_int("FETCHER_ROLLING_WINDOW_SIZE", 4000))
ROLLING_WINDOW_STEP = max(1, _env_int("FETCHER_ROLLING_WINDOW_STEP", 2000))
ROLLING_WINDOW_MAX_WINDOWS = max(0, _env_int("FETCHER_ROLLING_WINDOW_MAX_WINDOWS", 0))
_SPACY_SENTENCIZER = None


def _sanity_check_download_mode_config(
    mode: str,
    window_size: int,
    window_step: int,
) -> None:
    if mode not in {"text", "download_only", "rolling_extract"}:
        raise ValueError(f"Unsupported FETCHER_DOWNLOAD_MODE: {mode}")
    if window_size <= 0:
        raise ValueError("FETCHER_ROLLING_WINDOW_SIZE must be positive")
    if window_step <= 0:
        raise ValueError("FETCHER_ROLLING_WINDOW_STEP must be positive")
    if mode == "rolling_extract" and window_step > window_size:
        raise ValueError("FETCHER_ROLLING_WINDOW_STEP cannot exceed FETCHER_ROLLING_WINDOW_SIZE")


_sanity_check_download_mode_config(DOWNLOAD_MODE, ROLLING_WINDOW_SIZE, ROLLING_WINDOW_STEP)


@dataclass(frozen=True, slots=True)
class Validator:
    concurrency: int = 2
    per_domain: int = 1
    max_attempts: int = 2
    timeout: Optional[float] = None


@dataclass(frozen=True, slots=True)
class DebugTuning:
    brave_top_n: int = 0
    log_rule_hits: bool = True


# Central policy object to reduce hardcoded strings in function bodies.
@dataclass(frozen=True, slots=True)
class FetcherPolicy:
    paywall_domains: Set[str]
    paywall_status_codes: Set[int]
    paywall_hints: Tuple[str, ...]
    brave_excluded_domains: Set[str]
    overrides_path: Path
    local_sources_dir: Path
    strip_subdomains: frozenset[str] = frozenset({"www"})
    normalize_idna: bool = True
    enable_perplexity_resolver: bool = False
    overrides_reload: bool = False
    paywall_hints_path: Optional[Path] = None
    validator: Validator = Validator()
    debug: DebugTuning = DebugTuning()
    use_wayback_on_404: bool = False
    text_inline_max_bytes: int = 0


DEFAULT_POLICY = FetcherPolicy(
    paywall_domains=_normalize_set(PAYWALL_DOMAINS, {"www"}),
    paywall_status_codes={c for c in PAYWALL_STATUS_CODES if c != 404},
    paywall_hints=PAYWALL_HINTS,
    brave_excluded_domains=_normalize_set(BRAVE_EXCLUDED_DOMAINS, {"www"}),
    overrides_path=OVERRIDES_PATH,
    local_sources_dir=LOCAL_SOURCES_DIR,
    text_inline_max_bytes=TEXT_INLINE_MAX_BYTES,
)


_POLICY_LOCK = threading.Lock()

# ---------------- Single event loop helper for this module ------------------
_FETCH_LOOP: asyncio.AbstractEventLoop | None = None

def _run_in_fetch_loop(coro: "asyncio.coroutines.Coroutine"):
    global _FETCH_LOOP
    if _FETCH_LOOP is None or _FETCH_LOOP.is_closed():
        _FETCH_LOOP = asyncio.new_event_loop()
    try:
        return _FETCH_LOOP.run_until_complete(coro)
    except Exception:
        try:
            _FETCH_LOOP.close()
        except Exception:
            pass
        _FETCH_LOOP = asyncio.new_event_loop()
        return _FETCH_LOOP.run_until_complete(coro)


def set_overrides_path(path: str | Path) -> None:
    """Override the path used for overrides.json at runtime.

    Safe: only affects how _load_override_rules() is pointed; all other policy
    fields remain unchanged.
    """
    global DEFAULT_POLICY
    p = Path(path)
    # Guard against concurrent mutation; intended for startup-time use only
    with _POLICY_LOCK:
        DEFAULT_POLICY = FetcherPolicy(
            paywall_domains=set(DEFAULT_POLICY.paywall_domains),
            paywall_status_codes=set(DEFAULT_POLICY.paywall_status_codes),
            paywall_hints=DEFAULT_POLICY.paywall_hints,
            brave_excluded_domains=set(DEFAULT_POLICY.brave_excluded_domains),
            overrides_path=p,
            local_sources_dir=DEFAULT_POLICY.local_sources_dir,
        )


@dataclass(slots=True)
class FetcherResult:
    """Outcome payload returned to pipeline steps."""

    results: List[FetchResult]
    audit: Dict[str, Any]
    applied_alternates: List[Dict[str, Any]]
    outstanding_controls: int
    outstanding_urls: int



def _build_outstanding_reports(
    entries: Iterable[dict],
    results: List[FetchResult],
    run_artifacts_dir: Path,
) -> Tuple[int, int]:
    results_map: Dict[str, FetchResult] = {result.url: result for result in results}

    outstanding_controls: List[Dict[str, Any]] = []
    outstanding_urls: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        url = entry[K_URL].strip()
        result = results_map.get(url)
        status = result.status if result else -1
        text_ok = bool(result and result.status == 200 and result.text)
        if text_ok:
            continue

        failed_url = entry.get(K_ORIGINAL_URL, url)
        domain = entry.get(K_DOMAIN) or (result.domain if result else "")
        titles = entry.get(K_TITLES) or []
        worksheets = entry.get(K_WORKSHEETS) or []
        controls = entry.get(K_CONTROLS) or []

        url_summary = outstanding_urls.setdefault(
            failed_url,
            {
                K_URL: failed_url,
                K_DOMAIN: domain,
                K_STATUS: status,
                "control_ids": set(),
                "worksheets": set(worksheets),
            },
        )
        url_summary["worksheets"].update(worksheets)

        for idx, control_id in enumerate(controls):
            title = titles[idx] if idx < len(titles) else ""
            outstanding_controls.append(
                {
                    "control_id": control_id,
                    "title": title,
                    "failed_url": failed_url,
                    "domain": domain,
                    "worksheets": worksheets,
                    "status": status,
                }
            )
            url_summary["control_ids"].add(control_id)

    outstanding_controls.sort(
        key=lambda item: (item["domain"], item["failed_url"], item["control_id"])
    )

    url_rollup = []
    for data in outstanding_urls.values():
        url_rollup.append(
            {
                K_URL: data[K_URL],
                K_DOMAIN: data[K_DOMAIN],
                K_STATUS: data[K_STATUS],
                "control_count": len(data["control_ids"]),
                "worksheets": sorted(data["worksheets"]),
            }
        )

    url_rollup.sort(key=lambda item: (item[K_DOMAIN], item[K_URL]))

    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (run_artifacts_dir / "outstanding_controls_remaining.json").write_text(
        json.dumps(outstanding_controls, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_artifacts_dir / "outstanding_urls_summary.json").write_text(
        json.dumps(url_rollup, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return len(outstanding_controls), len(url_rollup)



def _build_outstanding_reports(
    entries: Iterable[dict],
    results: List[FetchResult],
    run_artifacts_dir: Path,
) -> Tuple[int, int]:
    results_map: Dict[str, FetchResult] = {result.url: result for result in results}

    outstanding_controls: List[Dict[str, Any]] = []
    outstanding_urls: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        url = entry[K_URL].strip()
        result = results_map.get(url)
        status = result.status if result else -1
        text_ok = bool(result and result.status == 200 and result.text)
        if text_ok:
            continue

        failed_url = entry.get(K_ORIGINAL_URL, url)
        domain = entry.get(K_DOMAIN) or (result.domain if result else "")
        titles = entry.get(K_TITLES) or []
        worksheets = entry.get(K_WORKSHEETS) or []
        controls = entry.get(K_CONTROLS) or []

        url_summary = outstanding_urls.setdefault(
            failed_url,
            {
                K_URL: failed_url,
                K_DOMAIN: domain,
                K_STATUS: status,
                "control_ids": set(),
                "worksheets": set(worksheets),
            },
        )
        url_summary["worksheets"].update(worksheets)

        for idx, control_id in enumerate(controls):
            title = titles[idx] if idx < len(titles) else ""
            outstanding_controls.append(
                {
                    "control_id": control_id,
                    "title": title,
                    "failed_url": failed_url,
                    "domain": domain,
                    "worksheets": worksheets,
                    "status": status,
                }
            )
            url_summary["control_ids"].add(control_id)

    outstanding_controls.sort(
        key=lambda item: (item["domain"], item["failed_url"], item["control_id"])
    )

    url_rollup = []
    for data in outstanding_urls.values():
        url_rollup.append(
            {
                K_URL: data[K_URL],
                K_DOMAIN: data[K_DOMAIN],
                K_STATUS: data[K_STATUS],
                "control_count": len(data["control_ids"]),
                "worksheets": sorted(data["worksheets"]),
            }
        )

    url_rollup.sort(key=lambda item: (item[K_DOMAIN], item[K_URL]))

    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    (run_artifacts_dir / "outstanding_controls_remaining.json").write_text(
        json.dumps(outstanding_controls, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_artifacts_dir / "outstanding_urls_summary.json").write_text(
        json.dumps(url_rollup, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return len(outstanding_controls), len(url_rollup)



def run_fetch_pipeline(
    entries: List[Dict[str, Any]],
    fetch_config: FetchConfig,
    inventory_path: Path,
    output_path: Path,
    audit_path: Path,
    cache_path: Optional[Path],
    run_artifacts_dir: Path,
    resolver_limit: int = 24,
    enable_resolver: bool = True,
    policy: FetcherPolicy = DEFAULT_POLICY,
    progress_hook: Optional[Callable[[int, int], None]] = None,
) -> FetcherResult:
    """Primary entrypoint used by pipeline scripts."""

    fetcher = URLFetcher(fetch_config, cache_path=cache_path)
    results, audit = _run_in_fetch_loop(fetcher.fetch_many(entries, progress_hook=progress_hook))

    annotate_paywall_metadata(results, policy)

    repo_root = resolve_repo_root(inventory_path)

    applied: List[Dict[str, Any]] = []
    if enable_resolver and entries:
        applied = resolve_paywalled_entries(
            entries,
            results,
            fetch_config,
            repo_root,
            inventory_path,
            run_artifacts_dir,
            limit=resolver_limit,
            policy=policy,
            run_fetch_loop=_run_in_fetch_loop,
        )

    success = sum(1 for r in results if r.status == 200 and r.text)
    failed = sum(1 for r in results if r.status != 200)

    outstanding_controls, outstanding_urls = _build_outstanding_reports(
        entries,
        results,
        run_artifacts_dir,
    )

    audit_payload = {
        **audit,
        "success": success,
        "failed": failed,
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    maybe_externalize_text(results, run_artifacts_dir, policy.text_inline_max_bytes)
    apply_download_mode(
        results,
        run_artifacts_dir,
        DOWNLOAD_MODE,
        ROLLING_WINDOW_SIZE,
        ROLLING_WINDOW_STEP,
        ROLLING_WINDOW_MAX_WINDOWS,
    )

    write_results(results, output_path)

    return FetcherResult(
        results=results,
        audit=audit_payload,
        applied_alternates=applied,
        outstanding_controls=outstanding_controls,
        outstanding_urls=outstanding_urls,
    )


def deterministic_batch_check(
    entries: List[Dict[str, Any]],
    fetch_config: FetchConfig,
    inventory_path: Path,
    output_path: Path,
    audit_path: Path,
    cache_path: Optional[Path],
    run_artifacts_dir: Path,
    resolver_limit: int = 24,
    policy: FetcherPolicy = DEFAULT_POLICY,
) -> Dict[str, Any]:
    """Run a deterministic batch fetch+resolve and report outstanding items.

    This is intended for small verification batches (e.g., missing URLs
    surfaced by coverage) and returns a machine‑readable dict with counts and
    file locations. Exit handling is up to the caller (0 when all covered).

    Behavior:
    - Applies override‑first resolver rules (overrides.json) before paywall gates.
    - Uses validation fetches for any alternates before writing changes.
    - Writes standard audit/outstanding reports under run_artifacts_dir.
    """

    result = run_fetch_pipeline(
        entries=entries,
        fetch_config=fetch_config,
        inventory_path=inventory_path,
        output_path=output_path,
        audit_path=audit_path,
        cache_path=cache_path,
        run_artifacts_dir=run_artifacts_dir,
        resolver_limit=resolver_limit,
        enable_resolver=True,
        policy=policy,
    )

    payload = {
        "success": int(result.audit.get("success", 0)),
        "failed": int(result.audit.get("failed", 0)),
        "applied_alternates": result.applied_alternates,
        "outstanding_controls": int(result.outstanding_controls),
        "outstanding_urls": int(result.outstanding_urls),
        "outstanding_controls_path": str(run_artifacts_dir / "outstanding_controls_remaining.json"),
        "outstanding_urls_path": str(run_artifacts_dir / "outstanding_urls_summary.json"),
        "audit_path": str(audit_path),
        "output_path": str(output_path),
    }
    return payload


def fetch_url(
    url: str,
    *,
    fetch_config: Optional[FetchConfig] = None,
    cache_path: Optional[Path] = None,
    run_artifacts_dir: Optional[Path] = None,
) -> FetchResult:
    """Fetch a single URL using the shared pipeline defaults."""

    normalized = (url or "").strip()
    if not normalized:
        raise ValueError("url must be a non-empty string")

    entry = {
        K_URL: normalized,
        K_DOMAIN: urlparse(normalized).netloc,
        K_CONTROLS: [],
        K_TITLES: [],
        K_FRAMEWORKS: [],
        K_WORKSHEETS: [],
    }
    config = fetch_config or FetchConfig()
    fetcher = URLFetcher(config, cache_path=cache_path)
    results, _ = _run_in_fetch_loop(fetcher.fetch_many([entry]))
    if not results:
        raise RuntimeError("Fetcher returned no result for the provided URL")
    result = results[0]

    annotate_paywall_metadata([result], DEFAULT_POLICY)

    if DOWNLOAD_MODE != "text":
        artifacts_dir = run_artifacts_dir or Path(os.getenv("FETCHER_SINGLE_RUN_ARTIFACTS", "run")) / "artifacts"
        apply_download_mode(
            [result],
            artifacts_dir,
            DOWNLOAD_MODE,
            ROLLING_WINDOW_SIZE,
            ROLLING_WINDOW_STEP,
            ROLLING_WINDOW_MAX_WINDOWS,
        )
    else:
        result.raw_bytes = None

    return result


def _load_inventory_entries(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON line in {path}: {line[:80]}") from None
            if K_URL not in entry:
                raise ValueError(f"Inventory entry missing '{K_URL}': {entry}")
            entries.append(entry)
    return entries


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetcher CLI entrypoint")
    parser.add_argument("--url", help="Fetch a single URL and print the JSON result")
    parser.add_argument("--inventory", type=Path, help="JSONL inventory for batch runs")
    parser.add_argument("--output", type=Path, help="Destination for fetch results JSONL")
    parser.add_argument("--audit", type=Path, help="Path to write audit metadata")
    parser.add_argument(
        "--run-artifacts",
        type=Path,
        default=Path("run") / "artifacts",
        help="Directory for artifacts such as outstanding reports",
    )
    parser.add_argument("--cache-path", type=Path, help="Optional HTTP cache path")
    parser.add_argument("--timeout", type=float, help="Override request timeout (seconds)")
    parser.add_argument("--concurrency", type=int, help="Max concurrent fetches")
    parser.add_argument("--per-domain", type=int, help="Max concurrent fetches per domain")
    parser.add_argument("--resolver-limit", type=int, default=24, help="Resolver limit per batch")
    parser.add_argument("--no-resolver", action="store_true", help="Disable alternate resolvers")

    args = parser.parse_args(argv)

    if not args.url and not args.inventory:
        parser.error("Either --url or --inventory must be supplied")
    if args.url and args.inventory:
        parser.error("Use either --url or --inventory, not both")

    if args.url:
        config = FetchConfig()
        if args.timeout:
            config.timeout = args.timeout
        if args.concurrency:
            config.concurrency = args.concurrency
        if args.per_domain:
            config.per_domain = args.per_domain
        result = fetch_url(args.url, fetch_config=config, cache_path=args.cache_path)
        payload = result.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as fh:
                fh.write(result.to_json() + "\n")
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
        return 0

    inventory_path: Path = args.inventory
    if not inventory_path.exists():
        raise FileNotFoundError(inventory_path)
    entries = _load_inventory_entries(inventory_path)
    if not entries:
        parser.error(f"Inventory {inventory_path} did not contain any entries")

    fetch_config = FetchConfig()
    if args.timeout:
        fetch_config.timeout = args.timeout
    if args.concurrency:
        fetch_config.concurrency = args.concurrency
    if args.per_domain:
        fetch_config.per_domain = args.per_domain

    cache_path = args.cache_path
    output_path = args.output or inventory_path.with_suffix(".results.jsonl")
    audit_path = args.audit or inventory_path.with_suffix(".audit.json")
    run_artifacts_dir = args.run_artifacts

    result = run_fetch_pipeline(
        entries=entries,
        fetch_config=fetch_config,
        inventory_path=inventory_path,
        output_path=output_path,
        audit_path=audit_path,
        cache_path=cache_path,
        run_artifacts_dir=run_artifacts_dir,
        resolver_limit=args.resolver_limit,
        enable_resolver=not args.no_resolver,
    )

    summary = {
        "success": result.audit.get("success", 0),
        "failed": result.audit.get("failed", 0),
        "outstanding_controls": result.outstanding_controls,
        "outstanding_urls": result.outstanding_urls,
        "output_path": str(output_path),
        "audit_path": str(audit_path),
        "run_artifacts_dir": str(run_artifacts_dir),
    }
    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
