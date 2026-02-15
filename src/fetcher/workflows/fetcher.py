"""Reusable helpers for Sparta URL fetching and alternate resolution."""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import hashlib
import mimetypes
import os
import re
import sys
import time
from dataclasses import dataclass, replace
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set
from urllib.parse import urlparse

import requests

from dotenv import load_dotenv

from .web_fetch import (
    FetchConfig,
    FetchResult,
    URLFetcher,
    write_results,
    _normalize_url as _wf_normalize_url,
)
from .paywall_detector import detect_paywall
from .paywall_utils import resolve_paywalled_entries, reload_overrides_cache

load_dotenv(override=True)
from .fetcher_config import (
    BRAVE_ENDPOINT,
    HDR_ACCEPT,
    HDR_BRAVE_TOKEN,
    PAYWALL_DOMAINS,
    PAYWALL_STATUS_CODES,
    PAYWALL_HINTS,
    PAYWALL_SAFE_DOMAINS,
    PAYWALL_SAFE_SUFFIXES,
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
    build_failure_summary,
    has_text_payload,
    idna_normalize as _idna_normalize,
    is_safe_domain,
    normalize_domain as _normalize_domain,
    normalize_set as _normalize_set,
    collect_environment_warnings,
    resolve_repo_root,
    validate_url,
)
from .download_utils import (
    annotate_paywall_metadata,
    apply_download_mode,
    maybe_externalize_text,
    materialize_extracted_text,
    materialize_markdown,
)
from .extract_utils import evaluate_result_content
from ..monitoring import EventEmitter
from ..quality import BATCH_GATE
from .outstanding_utils import build_outstanding_reports
from .doctor import build_doctor_report, format_doctor_report


# ---------------------------------------------------------------------------
# Constants shared with the pipeline entrypoint.
# ---------------------------------------------------------------------------


BRAVE_ENDPOINT = BRAVE_ENDPOINT
HDR_ACCEPT = HDR_ACCEPT
HDR_BRAVE_TOKEN = HDR_BRAVE_TOKEN

PAYWALL_DOMAINS = PAYWALL_DOMAINS
PAYWALL_STATUS_CODES = PAYWALL_STATUS_CODES
PAYWALL_HINTS = PAYWALL_HINTS
PAYWALL_SAFE_DOMAINS = PAYWALL_SAFE_DOMAINS
BRAVE_EXCLUDED_DOMAINS = BRAVE_EXCLUDED_DOMAINS
LOCAL_SOURCES_DIR = LOCAL_SOURCES_DIR
OVERRIDES_PATH = OVERRIDES_PATH


def _env_int(name: str, default: int = 0) -> int:
    try:
        raw = os.getenv(name, "")
        return int(raw) if raw.strip() else default
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


TEXT_INLINE_MAX_BYTES = _env_int("FETCHER_TEXT_INLINE_MAX_BYTES", 0)


def _normalize_download_mode(value: str | None) -> str:
    mode = (value or "text").strip().lower()
    if mode not in {"text", "download_only", "rolling_extract"}:
        return "text"
    return mode


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


def _resolve_download_config(
    download_mode: Optional[str] = None,
    window_size: Optional[int] = None,
    window_step: Optional[int] = None,
    max_windows: Optional[int] = None,
) -> Tuple[str, int, int, int]:
    """Resolve download knobs using overrides or environment defaults."""

    mode = _normalize_download_mode(download_mode or os.getenv("FETCHER_DOWNLOAD_MODE"))
    resolved_window_size = window_size if window_size is not None else max(1, _env_int("FETCHER_ROLLING_WINDOW_SIZE", 4000))
    resolved_window_step = window_step if window_step is not None else max(1, _env_int("FETCHER_ROLLING_WINDOW_STEP", 2000))
    resolved_max_windows = max_windows if max_windows is not None else max(0, _env_int("FETCHER_ROLLING_WINDOW_MAX_WINDOWS", 0))

    _sanity_check_download_mode_config(mode, resolved_window_size, resolved_window_step)
    return mode, resolved_window_size, resolved_window_step, resolved_max_windows


# Validate defaults at import so misconfigured env fails fast without caching values.
_resolve_download_config()


def _etl_minimal_help() -> str:
    return """Fetcher ETL CLI

Usage:
  fetcher-etl --url <url> [--output <path>] [--audit <path>] [--run-artifacts <dir>]
  fetcher-etl --inventory <inventory.jsonl> [--output <path>] [--audit <path>] [--run-artifacts <dir>]
  fetcher-etl --manifest <urls.txt|-> [--output <path>] [--audit <path>] [--run-artifacts <dir>]
  fetcher-etl --doctor

Common options:
  --url            Fetch a single URL (prints JSON result).
  --inventory      JSONL inventory for batch runs.
  --manifest       Line-based URL manifest (one URL per line).
  --output         Results JSONL path.
  --audit          Audit JSON path.
  --run-artifacts  Directory for artifacts (default: run/artifacts).
  --out            Alias for --run-artifacts.
  --soft-fail      Exit 0 even when environment warnings are present.
  --dry-run        Validate inputs and policy without fetching.
  --metrics-json   Emit metrics JSON to a path or '-' for stdout.
  --print-metrics  Print compact metrics summary to stderr.

Discoverability:
  --help-full     Expanded help, env vars, artifacts.
  --find <query>  Search flags, env vars, artifacts.
"""


def _etl_help_full() -> str:
    return """Fetcher ETL CLI (full)

Modes:
  --url <url>             Fetch a single URL (prints JSON result).
  --inventory <path>      JSONL inventory for batch runs.
  --manifest <path|->     Line-based manifest (one URL per line).

Core flags:
  --output <path>         Results JSONL path (default: <inventory>.results.jsonl).
  --audit <path>          Audit JSON path (default: <inventory>.audit.json).
  --run-artifacts <dir>   Directory for artifacts (default: run/artifacts).
  --out <dir>             Alias for --run-artifacts.
  --cache-path <path>     Optional HTTP cache path override.
  --timeout <seconds>     Override request timeout.
  --concurrency <n>       Max concurrent fetches.
  --per-domain <n>        Max concurrent fetches per domain.
  --resolver-limit <n>    Max alternates resolved per batch.
  --no-resolver           Disable alternate resolvers.
  --no-http-cache         Disable HTTP cache read/write for this run.
  --no-fanout             Disable link-hub fan-out.

Download mode:
  --download-mode <mode>  text | download_only | rolling_extract
  --window-size <chars>   Rolling window size.
  --window-step <chars>   Rolling window hop.
  --max-windows <n>       Maximum rolling windows (0 = unlimited).

Diagnostics:
  --doctor        Print environment/dependency diagnostics and exit.
  --dry-run       Validate inputs and policy without fetching.

Metrics:
  --metrics-json <path|->
  --print-metrics

Important env vars (existing):
  BRAVE_API_KEY
  BRAVE_SEARCH_API_KEY
  CHUTES_API_BASE
  CHUTES_API_KEY
  CHUTES_TEXT_MODEL
  FETCHER_OVERRIDES_PATH
  FETCHER_HTTP_CACHE_DISABLE
  FETCHER_HTTP_CACHE_PATH
  FETCHER_HTTP_CACHE_DIR
  FETCHER_DOWNLOAD_MODE
  FETCHER_ROLLING_WINDOW_SIZE
  FETCHER_ROLLING_WINDOW_STEP
  FETCHER_ROLLING_WINDOW_MAX_WINDOWS
  FETCHER_TEXT_INLINE_MAX_BYTES
  FETCHER_TEXT_CACHE_DIR

Artifacts:
  <inventory>.results.jsonl   Fetch results
  <inventory>.audit.json      Audit + environment warnings
  run/artifacts/**            Downloads, markdown, outstanding reports, junk table
  metrics JSON (optional)     When --metrics-json is set

Troubleshooting:
  - Missing Brave or Playwright deps surface as environment warnings.
  - Use --dry-run to validate inventory input quickly.
"""


_ETL_FIND_INDEX = [
    ("command", "url", "Single-URL mode via --url."),
    ("command", "inventory", "Batch mode via --inventory."),
    ("command", "manifest", "Batch mode via --manifest (line-based URLs)."),
    ("flag", "--url", "Fetch a single URL (prints JSON result)."),
    ("flag", "--inventory", "JSONL inventory for batch runs."),
    ("flag", "--manifest", "Line-based URL manifest (one URL per line)."),
    ("flag", "--output", "Results JSONL path."),
    ("flag", "--audit", "Audit JSON path."),
    ("flag", "--run-artifacts", "Directory for artifacts."),
    ("flag", "--out", "Alias for --run-artifacts."),
    ("flag", "--cache-path", "Override HTTP cache path."),
    ("flag", "--timeout", "Override request timeout."),
    ("flag", "--concurrency", "Max concurrent fetches."),
    ("flag", "--per-domain", "Max concurrent fetches per domain."),
    ("flag", "--resolver-limit", "Resolver limit per batch."),
    ("flag", "--no-resolver", "Disable alternate resolvers."),
    ("flag", "--no-http-cache", "Disable HTTP cache read/write."),
    ("flag", "--no-fanout", "Disable link-hub fan-out."),
    ("flag", "--download-mode", "Override download mode."),
    ("flag", "--window-size", "Rolling window size."),
    ("flag", "--window-step", "Rolling window hop."),
    ("flag", "--max-windows", "Maximum rolling windows."),
    ("flag", "--metrics-json", "Emit metrics JSON to a path or '-' for stdout."),
    ("flag", "--print-metrics", "Print compact metrics summary to stderr."),
    ("flag", "--soft-fail", "Exit 0 even when environment warnings are present."),
    ("flag", "--doctor", "Print environment diagnostics and exit."),
    ("flag", "--dry-run", "Validate inputs and policy without fetching."),
    ("flag", "--help-full", "Expanded help, env vars, artifacts."),
    ("flag", "--find", "Search flags, env vars, artifacts."),
    ("env", "BRAVE_API_KEY", "Enable Brave alternates."),
    ("env", "BRAVE_SEARCH_API_KEY", "Enable Brave alternates."),
    ("env", "CHUTES_API_BASE", "SciLLM API base."),
    ("env", "CHUTES_API_KEY", "SciLLM API key."),
    ("env", "CHUTES_TEXT_MODEL", "SciLLM model."),
    ("env", "FETCHER_OVERRIDES_PATH", "Override overrides.json path."),
    ("env", "FETCHER_HTTP_CACHE_DISABLE", "Disable HTTP cache."),
    ("env", "FETCHER_HTTP_CACHE_PATH", "Override HTTP cache path."),
    ("env", "FETCHER_DOWNLOAD_MODE", "Default download mode."),
    ("artifact", "results.jsonl", "Fetch results."),
    ("artifact", "audit.json", "Audit metadata."),
    ("artifact", "run/artifacts/", "Downloads, markdown, outstanding reports."),
    ("artifact", "metrics.json", "Optional metrics output."),
]


def _run_etl_find(query: str) -> str:
    needle = (query or "").strip().lower()
    if not needle:
        return ""
    lines: List[str] = []
    for category, name, desc in _ETL_FIND_INDEX:
        haystack = f"{category} {name} {desc}".lower()
        if needle in haystack:
            lines.append(f"{category} {name} - {desc}")
    return "\n".join(lines)


def _parse_manifest_lines(lines: Iterable[str]) -> List[str]:
    urls: List[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        if any(ch.isspace() for ch in line):
            raise ValueError(f"Invalid manifest line (inline metadata not allowed): {raw_line.rstrip()}")
        urls.append(line)
    return urls


def _load_manifest_urls(path_or_dash: str) -> List[str]:
    if path_or_dash == "-":
        return _parse_manifest_lines(sys.stdin.read().splitlines())
    path = Path(path_or_dash)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return _parse_manifest_lines(path.read_text(encoding="utf-8").splitlines())


def _build_basic_entries(urls: Iterable[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for url in urls:
        parsed = urlparse(url)
        entries.append(
            {
                K_URL: url,
                K_DOMAIN: parsed.netloc,
                K_CONTROLS: [],
                K_TITLES: [],
                K_FRAMEWORKS: [],
                K_WORKSHEETS: [],
            }
        )
    return entries


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
    paywall_safe_domains: Set[str]
    paywall_safe_suffixes: Tuple[str, ...]
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
    emit_extracted_text: bool = True
    extracted_text_min_chars: int = 1
    emit_markdown: bool = False
    markdown_min_chars: int = 1
    emit_fit_markdown: bool = True
    fit_markdown_min_chars: int = 200


DEFAULT_POLICY = FetcherPolicy(
    paywall_domains=_normalize_set(PAYWALL_DOMAINS, {"www"}),
    paywall_status_codes={c for c in PAYWALL_STATUS_CODES if c != 404},
    paywall_hints=PAYWALL_HINTS,
    paywall_safe_domains=_normalize_set(PAYWALL_SAFE_DOMAINS, {"www"}),
    paywall_safe_suffixes=PAYWALL_SAFE_SUFFIXES,
    brave_excluded_domains=_normalize_set(BRAVE_EXCLUDED_DOMAINS, {"www"}),
    overrides_path=OVERRIDES_PATH,
    local_sources_dir=LOCAL_SOURCES_DIR,
    text_inline_max_bytes=TEXT_INLINE_MAX_BYTES,
    emit_extracted_text=_env_bool("FETCHER_EMIT_EXTRACTED_TEXT", "1"),
    extracted_text_min_chars=max(1, _env_int("FETCHER_EXTRACTED_TEXT_MIN_CHARS", 1)),
    emit_markdown=_env_bool("FETCHER_EMIT_MARKDOWN", "0"),
    markdown_min_chars=max(1, _env_int("FETCHER_MARKDOWN_MIN_CHARS", 1)),
    emit_fit_markdown=_env_bool("FETCHER_EMIT_FIT_MARKDOWN", "1"),
    fit_markdown_min_chars=max(1, _env_int("FETCHER_FIT_MARKDOWN_MIN_CHARS", 200)),
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
        DEFAULT_POLICY = replace(DEFAULT_POLICY, overrides_path=p)


def _compute_metrics(results: List[FetchResult], audit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    counts = {"total": len(results), "success": 0, "failed": 0}
    content_verdict_counts: Dict[str, int] = {}
    paywall_verdict_counts: Dict[str, int] = {}
    fallback_counts: Dict[str, int] = {}
    proxy_attempted = 0
    proxy_success = 0

    for result in results:
        status = getattr(result, "status", None)
        success = bool(status == 200 and has_text_payload(result))
        if success:
            counts["success"] += 1
        else:
            counts["failed"] += 1

        metadata = result.metadata or {}
        content_verdict = str(metadata.get("content_verdict") or "unknown")
        paywall_verdict = str(metadata.get("paywall_verdict") or "unknown")
        content_verdict_counts[content_verdict] = content_verdict_counts.get(content_verdict, 0) + 1
        paywall_verdict_counts[paywall_verdict] = paywall_verdict_counts.get(paywall_verdict, 0) + 1

        fetch_path = metadata.get("fetch_path") or []
        if isinstance(fetch_path, list):
            for step in fetch_path:
                if step == "aiohttp":
                    continue
                fallback_counts[step] = fallback_counts.get(step, 0) + 1

        if metadata.get("proxy_rotation_attempted"):
            proxy_attempted += 1
        if metadata.get("proxy_rotation_used"):
            proxy_success += 1

    rate_limit_metrics = (audit or {}).get("rate_limit_metrics") if audit else None
    return {
        "counts": counts,
        "content_verdict_counts": content_verdict_counts,
        "paywall_verdict_counts": paywall_verdict_counts,
        "fallback_counts": fallback_counts,
        "proxy_rotation": {"attempted": proxy_attempted, "success": proxy_success},
        "rate_limit_metrics": rate_limit_metrics if rate_limit_metrics is not None else None,
    }


def _print_metrics_summary(metrics: Dict[str, Any]) -> None:
    counts = metrics.get("counts") or {}
    fallback_counts = metrics.get("fallback_counts") or {}
    proxy_rotation = metrics.get("proxy_rotation") or {}
    summary = (
        "[fetcher] metrics total={total} success={success} failed={failed} "
        "fallbacks={fallbacks} proxy_attempted={proxy_attempted} proxy_success={proxy_success}"
    ).format(
        total=counts.get("total", 0),
        success=counts.get("success", 0),
        failed=counts.get("failed", 0),
        fallbacks=",".join(f"{k}:{v}" for k, v in sorted(fallback_counts.items())) or "none",
        proxy_attempted=proxy_rotation.get("attempted", 0),
        proxy_success=proxy_rotation.get("success", 0),
    )
    print(summary, file=sys.stderr)


def _write_junk_report(results: List[FetchResult], run_artifacts_dir: Optional[Path]) -> None:
    if run_artifacts_dir is None:
        return
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    junk_path = run_artifacts_dir / "junk_results.jsonl"
    summary_path = run_artifacts_dir / "junk_summary.json"
    table_path = run_artifacts_dir / "junk_table.md"
    counts: Dict[str, int] = {}
    domains: Dict[str, Dict[str, int]] = {}
    examples: Dict[str, List[Dict[str, Any]]] = {}
    total = 0
    try:
        max_rows = int(os.getenv("FETCHER_JUNK_TABLE_MAX_ROWS", "200"))
    except Exception:
        max_rows = 200
    try:
        max_examples = int(os.getenv("FETCHER_JUNK_EXAMPLES_PER_VERDICT", "8"))
    except Exception:
        max_examples = 8
    with junk_path.open("w", encoding="utf-8") as fh:
        for result in results:
            metadata = result.metadata or {}
            verdict = metadata.get("content_verdict")
            if not verdict or verdict == "ok":
                continue
            payload = result.to_dict()
            payload.pop("text", None)
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
            counts[verdict] = counts.get(verdict, 0) + 1
            domain = str(payload.get("domain") or result.domain or "")
            if domain:
                domains.setdefault(domain, {})
                domains[domain][verdict] = domains[domain].get(verdict, 0) + 1
            if max_examples > 0:
                bucket = examples.setdefault(str(verdict), [])
                if len(bucket) < max_examples:
                    bucket.append(
                        {
                            "url": payload.get("url"),
                            "domain": domain or None,
                            "status": payload.get("status"),
                            "method": payload.get("method"),
                            "junk_reason": payload.get("junk_reason"),
                            "content_reasons": payload.get("content_reasons"),
                        }
                    )
            total += 1
    summary = {
        "total": total,
        "counts_by_verdict": counts,
        "counts_by_domain": domains,
        "examples_by_verdict": examples,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Human-friendly triage view (bounded).
    try:
        rows = []
        for result in results:
            metadata = result.metadata or {}
            verdict = metadata.get("content_verdict")
            if not verdict or verdict == "ok":
                continue
            payload = result.to_dict()
            url = str(payload.get("url") or "")
            domain = str(payload.get("domain") or "")
            status = str(payload.get("status") or "")
            method = str(payload.get("method") or "")
            reasons = payload.get("content_reasons") or []
            if isinstance(reasons, list):
                reasons_txt = ", ".join(str(r) for r in reasons if r)
            else:
                reasons_txt = str(reasons)
            junk_reason = str(payload.get("junk_reason") or "")
            rows.append((str(verdict), status, domain, method, junk_reason, reasons_txt, url))
        rows.sort(key=lambda r: (r[0], r[2], r[1], r[6]))
        if max_rows > 0:
            rows = rows[:max_rows]
        header = "| verdict | status | domain | method | junk_reason | content_reasons | url |\n"
        sep = "|---|---:|---|---|---|---|---|\n"
        lines = [header, sep]
        for verdict, status, domain, method, junk_reason, reasons_txt, url in rows:
            reasons_cell = (reasons_txt[:180] + "…") if len(reasons_txt) > 180 else reasons_txt
            lines.append(
                f"| {verdict} | {status} | {domain} | {method} | {junk_reason} | {reasons_cell} | {url} |\n"
            )
        table_path.write_text("".join(lines), encoding="utf-8")
    except Exception:
        pass


def _collect_pdf_paywall_mismatches(results: Iterable[FetchResult]) -> List[Dict[str, Any]]:
    """Return URLs where PDF text was extracted but paywall signals remain.

    These are high-value for audits: they indicate cases where generic
    paywall/junk heuristics may be over-blocking long-form, publicly
    accessible PDFs that we successfully parsed.
    """

    mismatches: List[Dict[str, Any]] = []
    for result in results:
        metadata = result.metadata or {}
        if not metadata.get("pdf_text_extracted"):
            continue
        paywall_verdict = (metadata.get("paywall_verdict") or "").lower()
        content_verdict = (metadata.get("content_verdict") or "").lower()
        if paywall_verdict in {"likely", "maybe", "paywall"} or content_verdict == "paywall":
            mismatches.append(
                {
                    "url": result.url,
                    "domain": result.domain,
                    "status": result.status,
                    "paywall_verdict": paywall_verdict or None,
                    "paywall_score": metadata.get("paywall_score"),
                    "content_verdict": content_verdict or None,
                    "pdf_pages": metadata.get("pdf_pages"),
                    "pdf_characters": metadata.get("pdf_characters"),
                }
            )
    return mismatches


def _annotate_content_changes(results: List[FetchResult], cache_path: Path) -> None:
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    dirty = False
    for result in results:
        metadata = dict(result.metadata or {})
        normalized = metadata.get("url_normalized") or _wf_normalize_url(result.url)
        if not normalized:
            continue
        if result.text:
            payload = result.text.encode("utf-8")
            sample = result.text[:800]
        elif result.raw_bytes:
            payload = result.raw_bytes
            sample = ""
        else:
            continue
        sha = hashlib.sha256(payload).hexdigest()
        metadata["content_sha256"] = sha
        metadata["content_text_len"] = len(result.text or "")
        previous = cache.get(normalized)
        if previous:
            metadata["content_previous_sha256"] = previous.get("sha")
            metadata["content_previous_len"] = previous.get("length")
            metadata["content_previous_fetched_at"] = previous.get("fetched_at")
            metadata["content_changed"] = sha != previous.get("sha")
            prev_sample = previous.get("sample") or ""
            if result.text and prev_sample:
                ratio = difflib.SequenceMatcher(None, prev_sample, result.text[: len(prev_sample) or 1]).ratio()
                metadata["content_diff_ratio"] = round(ratio, 4)
        else:
            metadata["content_changed"] = True

        cache[normalized] = {
            "sha": sha,
            "length": len(result.text or ""),
            "fetched_at": result.fetched_at,
            "sample": sample,
        }
        result.metadata = metadata
        dirty = True

    if dirty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_change_feed(results: Iterable[FetchResult], artifacts_dir: Path) -> None:
    path = artifacts_dir / "changes.jsonl"
    entries: List[str] = []
    for result in results:
        metadata = result.metadata or {}
        if not metadata.get("content_changed"):
            continue
        record = {
            "url": result.url,
            "domain": result.domain,
            "fetched_at": result.fetched_at,
            "previous_fetched_at": metadata.get("content_previous_fetched_at"),
            "content_sha256": metadata.get("content_sha256"),
            "content_previous_sha256": metadata.get("content_previous_sha256"),
            "content_diff_ratio": metadata.get("content_diff_ratio"),
            "text_path": metadata.get(K_TEXT_PATH) or metadata.get("file_path"),
            "blob_path": metadata.get("blob_path"),
        }
        entries.append(json.dumps(record, ensure_ascii=False))

    if not entries:
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


@dataclass(slots=True)
class FetcherResult:
    """Outcome payload returned to pipeline steps."""

    results: List[FetchResult]
    audit: Dict[str, Any]
    applied_alternates: List[Dict[str, Any]]
    outstanding_controls: int
    outstanding_urls: int
    news_abstracts: int = 0

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
    download_mode: Optional[str] = None,
    rolling_window_size: Optional[int] = None,
    rolling_window_step: Optional[int] = None,
    rolling_window_max_windows: Optional[int] = None,
) -> FetcherResult:
    """Primary entrypoint used by pipeline scripts."""

    env_warnings = collect_environment_warnings()
    for warning in env_warnings:
        message = warning.get("message") or warning.get("code") or "environment warning"
        remedy = warning.get("remedy")
        if remedy:
            print(f"[fetcher] warning: {message} ({remedy})", file=sys.stderr)
        else:
            print(f"[fetcher] warning: {message}", file=sys.stderr)

    if os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1":
        fetch_config.disable_http_cache = True
        cache_path = None
    if fetch_config.overrides_path is None:
        # Env override path wins over policy default
        env_overrides = os.getenv("FETCHER_OVERRIDES_PATH")
        fetch_config.overrides_path = Path(env_overrides) if env_overrides else policy.overrides_path

    if fetch_config.screenshots_dir is None:
        fetch_config.screenshots_dir = run_artifacts_dir / "screenshots"

    fetcher = URLFetcher(fetch_config, cache_path=cache_path)


    # Prepare for incremental processing
    hash_cache_path = run_artifacts_dir / "fetch_cache" / "content_hashes.json"
    content_cache: Dict[str, Any] = {}
    if hash_cache_path.exists():
        try:
            content_cache = json.loads(hash_cache_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Ensure output directory exists and clear/create the output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Open in 'w' mode to clear it, then close. We'll append in the hook.
    # Actually, we can just keep it open.
    output_fh = output_path.open("w", encoding="utf-8")

    # Task Monitor state file
    monitor_state_path = run_artifacts_dir / ".batch_state.json"
    failed_count = 0
    consecutive_errors = 0

    # NDJSON event emitter for real-time monitoring
    emitter = EventEmitter("fetcher-etl", task_id=run_artifacts_dir.name)
    emitter.init(total=len(entries), description="ETL Pipeline")

    def streaming_hook(completed: int, total: int, entry: Dict[str, Any], result: FetchResult) -> None:
        nonlocal failed_count, consecutive_errors
        # 1. Post-process the single result
        evaluate_result_content(result)

        if result.status != 200:
            failed_count += 1

        # 2. Materialize artifacts
        materialize_extracted_text(
            [result],
            run_artifacts_dir,
            enabled=bool(getattr(policy, "emit_extracted_text", True)),
            min_chars=int(getattr(policy, "extracted_text_min_chars", 1) or 1),
        )
        materialize_markdown(
            [result],
            run_artifacts_dir,
            enabled=bool(getattr(policy, "emit_markdown", False)),
            min_chars=int(getattr(policy, "markdown_min_chars", 1) or 1),
            emit_fit_markdown=bool(getattr(policy, "emit_fit_markdown", True)),
            fit_min_chars=int(getattr(policy, "fit_markdown_min_chars", 200) or 200),
            overrides_path=getattr(policy, "overrides_path", None),
        )

        # 3. Paywall annotation
        annotate_paywall_metadata([result], policy)

        # 4. Content change detection (Incremental version of _annotate_content_changes)
        metadata = dict(result.metadata or {})
        normalized = metadata.get("url_normalized") or _wf_normalize_url(result.url)

        if normalized:
            payload: bytes = b""
            sample: str = ""
            if result.text:
                payload = result.text.encode("utf-8")
                sample = result.text[:800]
            elif result.raw_bytes:
                payload = result.raw_bytes
                sample = ""

            if payload:
                sha = hashlib.sha256(payload).hexdigest()
                metadata["content_sha256"] = sha
                metadata["content_text_len"] = len(result.text or "")
                previous = content_cache.get(normalized)
                if previous:
                    metadata["content_previous_sha256"] = previous.get("sha")
                    metadata["content_previous_len"] = previous.get("length")
                    metadata["content_previous_fetched_at"] = previous.get("fetched_at")
                    metadata["content_changed"] = sha != previous.get("sha")
                    prev_sample = previous.get("sample") or ""
                    if result.text and prev_sample:
                        ratio = difflib.SequenceMatcher(None, prev_sample, result.text[: len(prev_sample) or 1]).ratio()
                        metadata["content_diff_ratio"] = round(ratio, 4)
                else:
                    metadata["content_changed"] = True

                content_cache[normalized] = {
                    "sha": sha,
                    "length": len(result.text or ""),
                    "fetched_at": result.fetched_at,
                    "sample": sample,
                }
                result.metadata = metadata

        # 5. Write to output file immediately
        try:
            output_fh.write(result.to_json() + "\n")
            output_fh.flush()
        except Exception as e:
            print(f"[fetcher] error writing result: {e}", file=sys.stderr)

        # 6. Log progress
        status_icon = "✓" if result.status == 200 else "✗"
        print(f"[fetcher] {status_icon} {completed}/{total} {result.status} {result.url}", file=sys.stderr)

        # 6a. Emit NDJSON event for task-monitor integration
        emitter.fetch_complete(
            index=completed,
            url=result.url,
            status=result.status,
            ok=result.status == 200,
            domain=result.domain,
            method=result.method,
            content_verdict=(result.metadata or {}).get("content_verdict"),
        )

        # 6b. Track consecutive errors and check quality gate
        if result.status != 200:
            consecutive_errors += 1
        else:
            consecutive_errors = 0

        should_continue, reason = BATCH_GATE.check_batch_health(
            completed=completed,
            failed=failed_count,
            consecutive_errors=consecutive_errors,
        )
        if not should_continue:
            emitter.error(url="batch", error=f"Quality gate: {reason}", category="quality_gate")
            print(f"[fetcher] WARN: Quality gate triggered: {reason}", file=sys.stderr)

        # 7. Chain original hook
        if progress_hook:
            progress_hook(completed, total)

        # 8. Update task-monitor state
        try:
            state = {
                "completed": completed,
                "total": total,
                "description": f"Fetching URLs ({run_artifacts_dir.name})",
                "current_item": result.url[:100],
                "stats": {
                    "success": completed - failed_count,
                    "failed": failed_count
                },
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running"
            }
            monitor_state_path.write_text(json.dumps(state), encoding="utf-8")
        except Exception:
            pass

    try:
        results, audit = _run_in_fetch_loop(fetcher.fetch_many(entries, progress_hook=streaming_hook))
    finally:
        output_fh.close()

    # Emit completion event
    emitter.done(
        success=failed_count == 0,
        summary={"total": len(results), "failed": failed_count, "consecutive_errors": consecutive_errors},
    )

    # Save the updated cache back to disk
    if content_cache:
        hash_cache_path.parent.mkdir(parents=True, exist_ok=True)
        hash_cache_path.write_text(json.dumps(content_cache, ensure_ascii=False, indent=2), encoding="utf-8")

    # Final task-monitor update
    try:
        if monitor_state_path.exists():
            state = json.loads(monitor_state_path.read_text(encoding="utf-8"))
            state["status"] = "completed"
            state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
            monitor_state_path.write_text(json.dumps(state), encoding="utf-8")
    except Exception:
        pass


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

    success = sum(1 for r in results if r.status == 200 and has_text_payload(r))
    failed = sum(1 for r in results if r.status != 200)
    news_abstract_count = sum(1 for r in results if (r.metadata or {}).get("news_abstract"))

    outstanding_controls, outstanding_urls = build_outstanding_reports(
        entries,
        results,
        run_artifacts_dir,
        policy,
    )

    # Learn from failures if enabled
    if os.getenv("FETCHER_LEARN_FAILURES", "0") == "1":
        try:
            from ..memory_integration import learn_failure_patterns
            outstanding_path = run_artifacts_dir / "outstanding_urls_summary.json"
            if outstanding_path.exists():
                outstanding_data = json.loads(outstanding_path.read_text(encoding="utf-8"))
                failures = [
                    {
                        "url": item.get("url"),
                        "domain": item.get("domain"),
                        "category": item.get("category"),
                        "error": item.get("reason"),
                    }
                    for item in outstanding_data.get("items", [])
                ]
                learned = learn_failure_patterns(failures, scope="fetcher")
                if learned > 0:
                    print(f"[fetcher] Learned {learned} failure patterns to memory", file=sys.stderr)
        except Exception as e:
            print(f"[fetcher] Memory learning skipped: {e}", file=sys.stderr)

    pdf_paywall_mismatches = _collect_pdf_paywall_mismatches(results)

    audit_payload = {
        **audit,
        "success": success,
        "failed": failed,
    }
    if env_warnings:
        audit_payload["environment_warnings"] = env_warnings
        audit_payload["soft_failures"] = [item.get("code") for item in env_warnings if item.get("code")]
    if news_abstract_count:
        audit_payload["news_abstracts"] = news_abstract_count
    if pdf_paywall_mismatches:
        audit_payload["pdf_paywall_mismatches"] = pdf_paywall_mismatches
        audit_payload["pdf_paywall_mismatches_count"] = len(pdf_paywall_mismatches)
    audit_payload["failure_summary"] = build_failure_summary(results)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    maybe_externalize_text(results, run_artifacts_dir, policy.text_inline_max_bytes)
    resolved_mode, resolved_window_size, resolved_window_step, resolved_max_windows = _resolve_download_config(
        download_mode=download_mode,
        window_size=rolling_window_size,
        window_step=rolling_window_step,
        max_windows=rolling_window_max_windows,
    )
    apply_download_mode(
        results,
        run_artifacts_dir,
        resolved_mode,
        resolved_window_size,
        resolved_window_step,
        resolved_max_windows,
    )

    # _write_change_feed needs to run after results are populated
    _write_change_feed(results, run_artifacts_dir)

    # write_results(results, output_path)  <-- DISABLED: handled incrementally in streaming_hook
    _write_junk_report(results, run_artifacts_dir)

    return FetcherResult(
        results=results,
        audit=audit_payload,
        applied_alternates=applied,
        outstanding_controls=outstanding_controls,
        outstanding_urls=outstanding_urls,
        news_abstracts=news_abstract_count,
    )


def _build_dry_run_report(
    entries: List[Dict[str, Any]],
    *,
    policy: FetcherPolicy,
    inventory_path: Optional[Path],
    output_path: Optional[Path],
    audit_path: Optional[Path],
    run_artifacts_dir: Optional[Path],
    enable_resolver: bool,
) -> Dict[str, Any]:
    invalid_entries: List[Dict[str, str]] = []
    resolver_domain_candidates = 0
    safe_domains = 0

    for entry in entries:
        url = str(entry.get(K_URL, "")).strip()
        error = validate_url(url)
        if error:
            invalid_entries.append({"url": url, "error": error})
            continue
        domain = entry.get(K_DOMAIN) or urlparse(url).netloc
        normalized = _normalize_domain(str(domain or ""), policy.strip_subdomains)
        if is_safe_domain(
            normalized,
            policy.paywall_safe_domains,
            policy.paywall_safe_suffixes,
            policy.strip_subdomains,
        ):
            safe_domains += 1
            continue
        if normalized in policy.paywall_domains and enable_resolver:
            resolver_domain_candidates += 1

    overrides_path = Path(os.getenv("FETCHER_OVERRIDES_PATH") or policy.overrides_path)
    report: Dict[str, Any] = {
        "dry_run": True,
        "counts": {
            "total": len(entries),
            "invalid": len(invalid_entries),
            "resolver_domain_candidates": resolver_domain_candidates,
            "safe_domains": safe_domains,
        },
        "invalid_entries": invalid_entries,
        "policy_summary": {
            "paywall_domains": len(policy.paywall_domains),
            "paywall_safe_domains": len(policy.paywall_safe_domains),
            "overrides_path": str(overrides_path),
            "overrides_exists": overrides_path.exists(),
            "resolver_enabled": bool(enable_resolver),
        },
    }
    if inventory_path is not None:
        report["inventory_path"] = str(inventory_path)
    if output_path is not None:
        report["output_path"] = str(output_path)
    if audit_path is not None:
        report["audit_path"] = str(audit_path)
    if run_artifacts_dir is not None:
        report["run_artifacts_dir"] = str(run_artifacts_dir)
    return report


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
    download_mode: Optional[str] = None,
    rolling_window_size: Optional[int] = None,
    rolling_window_step: Optional[int] = None,
    rolling_window_max_windows: Optional[int] = None,
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

    if os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1":
        fetch_config.disable_http_cache = True
        cache_path = None
    if fetch_config.overrides_path is None:
        env_overrides = os.getenv("FETCHER_OVERRIDES_PATH")
        fetch_config.overrides_path = Path(env_overrides) if env_overrides else policy.overrides_path

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
        download_mode=download_mode,
        rolling_window_size=rolling_window_size,
        rolling_window_step=rolling_window_step,
        rolling_window_max_windows=rolling_window_max_windows,
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
    download_mode: Optional[str] = None,
    rolling_window_size: Optional[int] = None,
    rolling_window_step: Optional[int] = None,
    rolling_window_max_windows: Optional[int] = None,
) -> FetchResult:
    """Fetch a single URL using the shared pipeline defaults."""

    env_warnings = collect_environment_warnings()
    for warning in env_warnings:
        message = warning.get("message") or warning.get("code") or "environment warning"
        remedy = warning.get("remedy")
        if remedy:
            print(f"[fetcher] warning: {message} ({remedy})", file=sys.stderr)
        else:
            print(f"[fetcher] warning: {message}", file=sys.stderr)

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
    if os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1":
        config.disable_http_cache = True
    # Single-URL calls should not unexpectedly fan out into hub children.
    if getattr(config, "enable_toc_fanout", None) is None:
        config.enable_toc_fanout = False
    artifacts_dir = run_artifacts_dir or Path(os.getenv("FETCHER_SINGLE_RUN_ARTIFACTS", "run")) / "artifacts"
    if config.screenshots_dir is None:
        config.screenshots_dir = artifacts_dir / "screenshots"
    fetcher = URLFetcher(config, cache_path=cache_path)
    results, _ = _run_in_fetch_loop(fetcher.fetch_many([entry]))
    if not results:
        raise RuntimeError("Fetcher returned no result for the provided URL")
    result = results[0]

    # Apply the same content-quality gate used in batch runs so single-URL
    # fetches respect paywall/content heuristics consistently.
    evaluate_result_content(result)

    materialize_extracted_text(
        [result],
        artifacts_dir,
        enabled=bool(getattr(DEFAULT_POLICY, "emit_extracted_text", True)),
        min_chars=int(getattr(DEFAULT_POLICY, "extracted_text_min_chars", 1) or 1),
    )
    materialize_markdown(
        [result],
        artifacts_dir,
        enabled=bool(getattr(DEFAULT_POLICY, "emit_markdown", False)),
        min_chars=int(getattr(DEFAULT_POLICY, "markdown_min_chars", 1) or 1),
        emit_fit_markdown=bool(getattr(DEFAULT_POLICY, "emit_fit_markdown", True)),
        fit_min_chars=int(getattr(DEFAULT_POLICY, "fit_markdown_min_chars", 200) or 200),
        overrides_path=getattr(DEFAULT_POLICY, "overrides_path", None),
    )

    annotate_paywall_metadata([result], DEFAULT_POLICY)

    if env_warnings:
        metadata = dict(result.metadata or {})
        metadata["environment_warnings"] = env_warnings
        result.metadata = metadata

    resolved_mode, resolved_window_size, resolved_window_step, resolved_max_windows = _resolve_download_config(
        download_mode=download_mode,
        window_size=rolling_window_size,
        window_step=rolling_window_step,
        max_windows=rolling_window_max_windows,
    )

    if resolved_mode != "text":
        apply_download_mode(
            [result],
            artifacts_dir,
            resolved_mode,
            resolved_window_size,
            resolved_window_step,
            resolved_max_windows,
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
    parser = argparse.ArgumentParser(description="Fetcher CLI entrypoint", add_help=False)
    parser.add_argument("--help", "-h", action="store_true", help="Show minimal help")
    parser.add_argument("--help-full", action="store_true", help="Show expanded help")
    parser.add_argument("--find", help="Search flags, env vars, artifacts")
    parser.add_argument("--url", help="Fetch a single URL and print the JSON result")
    parser.add_argument("--inventory", type=Path, help="JSONL inventory for batch runs")
    parser.add_argument("--manifest", help="Line-based manifest (one URL per line)")
    parser.add_argument("--output", type=Path, help="Destination for fetch results JSONL")
    parser.add_argument("--audit", type=Path, help="Path to write audit metadata")
    parser.add_argument("--doctor", action="store_true", help="Print environment diagnostics and exit")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and policy without fetching")
    parser.add_argument(
        "--run-artifacts",
        type=Path,
        default=Path("run") / "artifacts",
        help="Directory for artifacts such as outstanding reports",
    )
    parser.add_argument("--out", type=Path, help="Alias for --run-artifacts")
    parser.add_argument("--cache-path", type=Path, help="Optional HTTP cache path")
    parser.add_argument("--timeout", type=float, help="Override request timeout (seconds)")
    parser.add_argument("--concurrency", type=int, help="Max concurrent fetches")
    parser.add_argument("--per-domain", type=int, help="Max concurrent fetches per domain")
    parser.add_argument("--resolver-limit", type=int, default=24, help="Resolver limit per batch")
    parser.add_argument("--no-resolver", action="store_true", help="Disable alternate resolvers")
    parser.add_argument("--no-http-cache", action="store_true", help="Disable HTTP cache read/write for this run")
    parser.add_argument(
        "--soft-fail",
        action="store_true",
        help="Exit 0 even when environment warnings are present (missing Brave/Playwright).",
    )
    parser.add_argument("--metrics-json", help="Emit metrics JSON to a path or '-' for stdout")
    parser.add_argument("--print-metrics", action="store_true", help="Print compact metrics summary to stderr")
    parser.add_argument(
        "--no-fanout",
        action="store_true",
        help="Disable link-hub fan-out (Phase 2) for this run",
    )
    parser.add_argument("--download-mode", choices=["text", "download_only", "rolling_extract"], help="Override download mode for this run")
    parser.add_argument("--window-size", type=int, help="Rolling window size (characters)")
    parser.add_argument("--window-step", type=int, help="Rolling window hop (characters)")
    parser.add_argument("--max-windows", type=int, help="Maximum rolling windows to emit (0 for unlimited)")

    args = parser.parse_args(argv)

    if args.help_full:
        sys.stdout.write(_etl_help_full())
        return 0
    if args.find is not None:
        output = _run_etl_find(args.find)
        if output:
            sys.stdout.write(output + "\n")
        return 0
    if args.help:
        sys.stdout.write(_etl_minimal_help())
        return 0
    if args.doctor:
        report = build_doctor_report(overrides_path=DEFAULT_POLICY.overrides_path)
        sys.stdout.write(format_doctor_report(report))
        return 0 if report.get("ok", True) else 2

    if not args.url and not args.inventory and not args.manifest:
        parser.error("Either --url, --inventory, or --manifest must be supplied")
    mode_count = sum(1 for item in (args.url, args.inventory, args.manifest) if item)
    if mode_count > 1:
        parser.error("Use only one of --url, --inventory, or --manifest")

    run_artifacts_dir = args.out or args.run_artifacts

    if args.url:
        if args.dry_run:
            entries = [{K_URL: args.url}]
            report = _build_dry_run_report(
                entries,
                policy=DEFAULT_POLICY,
                inventory_path=None,
                output_path=args.output,
                audit_path=args.audit,
                run_artifacts_dir=run_artifacts_dir,
                enable_resolver=not args.no_resolver,
            )
            sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            return 0 if report["counts"]["invalid"] == 0 else 3
        config = FetchConfig()
        if args.timeout:
            config.timeout = args.timeout
        if args.concurrency:
            config.concurrency = args.concurrency
        if args.per_domain:
            config.per_domain = args.per_domain
        config.disable_http_cache = args.no_http_cache or (os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1")
        if args.no_fanout:
            config.enable_toc_fanout = False
        config.overrides_path = DEFAULT_POLICY.overrides_path
        result = fetch_url(
            args.url,
            fetch_config=config,
            cache_path=args.cache_path,
            download_mode=args.download_mode,
            rolling_window_size=args.window_size,
            rolling_window_step=args.window_step,
            rolling_window_max_windows=args.max_windows,
        )
        payload = result.to_dict()
        env_warnings = (result.metadata or {}).get("environment_warnings") or []
        if env_warnings:
            payload["soft_failures"] = [item.get("code") for item in env_warnings if item.get("code")]
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as fh:
                fh.write(result.to_json() + "\n")
        metrics = None
        if args.metrics_json or args.print_metrics:
            metrics = _compute_metrics([result], audit=None)
        metrics_to_stdout = args.metrics_json == "-"
        if args.metrics_json:
            if metrics is None:
                metrics = _compute_metrics([result], audit=None)
            if metrics_to_stdout:
                json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
                sys.stdout.write("\n")
            else:
                metrics_path = Path(args.metrics_json)
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.print_metrics:
            _print_metrics_summary(metrics or _compute_metrics([result], audit=None))
        if not metrics_to_stdout:
            json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        if env_warnings and not args.soft_fail:
            return 3
        return 0

    inventory_path: Optional[Path] = None
    entries: List[Dict[str, Any]] = []
    if args.inventory:
        inventory_path = args.inventory
        if not inventory_path.exists():
            raise FileNotFoundError(inventory_path)
        entries = _load_inventory_entries(inventory_path)
    elif args.manifest:
        urls = _load_manifest_urls(args.manifest)
        entries = _build_basic_entries(urls)
        if args.manifest != "-":
            inventory_path = Path(args.manifest)
        else:
            inventory_path = Path.cwd() / "manifest.stdin.jsonl"

    if not entries:
        if inventory_path:
            parser.error(f"Inventory {inventory_path} did not contain any entries")
        parser.error("Manifest did not contain any entries")
    if args.dry_run:
        report = _build_dry_run_report(
            entries,
            policy=DEFAULT_POLICY,
            inventory_path=inventory_path,
            output_path=args.output,
            audit_path=args.audit,
            run_artifacts_dir=run_artifacts_dir,
            enable_resolver=not args.no_resolver,
        )
        sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        return 0 if report["counts"]["invalid"] == 0 else 3

    fetch_config = FetchConfig()
    if args.timeout:
        fetch_config.timeout = args.timeout
    if args.concurrency:
        fetch_config.concurrency = args.concurrency
    if args.per_domain:
        fetch_config.per_domain = args.per_domain
    fetch_config.disable_http_cache = args.no_http_cache or (os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1")
    if args.no_fanout:
        fetch_config.enable_toc_fanout = False
    fetch_config.overrides_path = DEFAULT_POLICY.overrides_path

    cache_path = args.cache_path
    if inventory_path is None:
        inventory_path = run_artifacts_dir / "manifest.jsonl"
    output_path = args.output or inventory_path.with_suffix(".results.jsonl")
    audit_path = args.audit or inventory_path.with_suffix(".audit.json")

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
        download_mode=args.download_mode,
        rolling_window_size=args.window_size,
        rolling_window_step=args.window_step,
        rolling_window_max_windows=args.max_windows,
    )

    metrics = None
    if args.metrics_json or args.print_metrics:
        metrics = _compute_metrics(result.results, audit=result.audit)

    summary = {
        "success": result.audit.get("success", 0),
        "failed": result.audit.get("failed", 0),
        "outstanding_controls": result.outstanding_controls,
        "outstanding_urls": result.outstanding_urls,
        "output_path": str(output_path),
        "audit_path": str(audit_path),
        "run_artifacts_dir": str(run_artifacts_dir),
    }
    env_warnings = result.audit.get("environment_warnings") or []
    if env_warnings:
        summary["environment_warnings"] = env_warnings
        summary["soft_failures"] = [item.get("code") for item in env_warnings if item.get("code")]
    metrics_to_stdout = args.metrics_json == "-"
    if args.metrics_json:
        if metrics is None:
            metrics = _compute_metrics(result.results, audit=result.audit)
        if metrics_to_stdout:
            json.dump(metrics, sys.stdout, ensure_ascii=False, indent=2)
            sys.stdout.write("\n")
        else:
            metrics_path = Path(args.metrics_json)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_metrics:
        _print_metrics_summary(metrics or _compute_metrics(result.results, audit=result.audit))
    if not metrics_to_stdout:
        json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    if env_warnings and not args.soft_fail:
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
