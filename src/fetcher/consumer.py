from __future__ import annotations

import json
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, TextIO, Tuple
from urllib.parse import urlparse

from .core.keys import K_CONTROLS, K_DOMAIN, K_FRAMEWORKS, K_TITLES, K_URL, K_WORKSHEETS
from .workflows.download_utils import annotate_paywall_metadata, materialize_extracted_text, materialize_markdown, persist_downloads
from .workflows.extract_utils import evaluate_result_content
from .workflows.fetcher import DEFAULT_POLICY, _run_in_fetch_loop, _env_bool, _env_int
from .workflows.fetcher_utils import collect_environment_warnings
from .workflows.web_fetch import FetchConfig, FetchResult, URLFetcher

CONSUMER_ID_KEY = "consumer_id"
CONSUMER_ORIGINAL_URL_KEY = "consumer_original_url"


def parse_manifest_lines(lines: Iterable[str]) -> List[str]:
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


def load_manifest(path_or_dash: str, *, stdin: Optional[TextIO] = None) -> List[str]:
    if path_or_dash == "-":
        stream = stdin or sys.stdin
        return parse_manifest_lines(stream.read().splitlines())
    path = Path(path_or_dash)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return parse_manifest_lines(path.read_text(encoding="utf-8").splitlines())


def generate_run_id(now: Optional[datetime] = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    suffix = secrets.token_hex(3)
    return f"{stamp}_{suffix}"


def resolve_run_dir(out_dir: Optional[Path]) -> Tuple[Path, str]:
    run_id = generate_run_id()
    if out_dir:
        return out_dir, run_id
    return Path("run") / "artifacts" / run_id, run_id


def resolve_cache_path() -> Optional[Path]:
    if os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1":
        return None
    env_path = os.getenv("FETCHER_HTTP_CACHE_PATH")
    if env_path:
        return Path(env_path)
    base = Path(os.getenv("FETCHER_HTTP_CACHE_DIR", "run/fetch_cache"))
    return base / "http_cache.json"


def parse_emit_csv(value: str) -> Set[str]:
    raw = (value or "").strip()
    if not raw:
        return set()
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    return set(tokens)


def _build_entries(urls: Sequence[str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, url in enumerate(urls):
        parsed = urlparse(url)
        entry = {
            K_URL: url,
            K_DOMAIN: parsed.netloc,
            K_CONTROLS: [],
            K_TITLES: [],
            K_FRAMEWORKS: [],
            K_WORKSHEETS: [],
            CONSUMER_ID_KEY: idx,
            CONSUMER_ORIGINAL_URL_KEY: url,
        }
        entries.append(entry)
    return entries


def _derive_paywall_verdict(result: FetchResult) -> Optional[str]:
    metadata = result.metadata or {}
    verdict = metadata.get("paywall_verdict")
    if verdict is None:
        detection = metadata.get("paywall_detection") or {}
        verdict = detection.get("verdict")
    if verdict is None:
        return None
    verdict = str(verdict).lower()
    if verdict in {"safe", "unlikely", "ok"}:
        return "ok"
    if verdict in {"maybe", "likely", "paywall", "paywalled"}:
        return "paywalled"
    return "unknown"


def _derive_verdict(result: FetchResult) -> str:
    metadata = result.metadata or {}
    content_verdict = (metadata.get("content_verdict") or "").lower()
    if result.error or result.status != 200 or result.method == "error":
        return "failed"
    if content_verdict and content_verdict != "ok":
        return "junk"
    return "ok"


def _derive_alternate_provider(result: FetchResult) -> Optional[str]:
    metadata = result.metadata or {}
    via = (metadata.get("via") or "").lower()
    if via in {"wayback", "jina"}:
        return via
    source = (metadata.get("source") or "").lower()
    if source in {"brave_search", "brave"}:
        return "brave"
    if metadata.get("alternate_note") == "brave_redirect":
        return "brave"
    return None


def _derive_final_url(result: FetchResult) -> str:
    metadata = result.metadata or {}
    for key in ("final_url", "final_downloaded_url"):
        value = metadata.get(key)
        if value:
            return str(value)
    for key in ("github_fetch_url", "jina_url", "google_translate_url"):
        value = metadata.get(key)
        if value:
            return str(value)
    return result.url


def _build_consumer_summary(
    command: str,
    run_id: str,
    run_dir: Path,
    started_at: datetime,
    finished_at: datetime,
    entries: Sequence[Dict[str, Any]],
    results: Sequence[FetchResult],
    emit: Set[str],
    emit_fit_md: bool,
    fit_md_requested: bool,
    download_errors: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    results_by_id: Dict[Any, FetchResult] = {}
    for result in results:
        cid = (result.metadata or {}).get(CONSUMER_ID_KEY)
        if cid is not None:
            results_by_id[cid] = result

    items: List[Dict[str, Any]] = []
    for entry in entries:
        cid = entry.get(CONSUMER_ID_KEY)
        original_url = entry.get(CONSUMER_ORIGINAL_URL_KEY) or entry.get(K_URL)
        requested_url = entry.get(K_URL)
        result = results_by_id.get(cid)
        warnings: List[str] = []
        errors: List[str] = []
        if result is None:
            errors.append("missing_result")
            item = {
                "original_url": original_url,
                "requested_url": requested_url,
                "final_downloaded_url": requested_url,
                "status": -1,
                "content_type": None,
                "method": "error",
                "alternate_provider": None,
                "verdict": "failed",
                "paywall_verdict": None,
                "artifacts": {
                    "download_path": None,
                    "extracted_text_path": None,
                    "markdown_path": None,
                    "fit_markdown_path": None,
                },
                "warnings": warnings,
                "errors": errors,
            }
            items.append(item)
            continue

        metadata = result.metadata or {}
        download_path = metadata.get("blob_path") or metadata.get("file_path")
        extracted_text_path = metadata.get("extracted_text_path")
        markdown_path = metadata.get("markdown_path")
        fit_markdown_path = metadata.get("fit_markdown_path")
        if "download" in emit and not download_path:
            if metadata.get("download_missing_no_payload"):
                warnings.append("download_missing_no_payload")
            else:
                error_value = None
                if download_errors is not None:
                    error_value = download_errors.get(result.url)
                error_value = error_value or metadata.get("download_error")
                if error_value:
                    errors.append(error_value)
                else:
                    errors.append("download_not_persisted")
        if "text" in emit and not extracted_text_path:
            warnings.append("text_not_materialized")
        if "md" in emit and not markdown_path:
            warnings.append("markdown_not_materialized")
        if fit_md_requested and "md" not in emit:
            warnings.append("fit_md_requested_without_md")
        if emit_fit_md and "md" in emit and not fit_markdown_path:
            warnings.append("fit_markdown_not_materialized")
        if result.error:
            errors.append(result.error)

        item = {
            "original_url": original_url,
            "requested_url": requested_url,
            "final_downloaded_url": _derive_final_url(result),
            "status": result.status,
            "content_type": result.content_type,
            "method": result.method,
            "alternate_provider": _derive_alternate_provider(result),
            "verdict": _derive_verdict(result),
            "paywall_verdict": _derive_paywall_verdict(result),
            "artifacts": {
                "download_path": download_path,
                "extracted_text_path": extracted_text_path,
                "markdown_path": markdown_path,
                "fit_markdown_path": fit_markdown_path,
            },
            "warnings": warnings,
            "errors": errors,
        }
        items.append(item)

    results_for_items = list(results_by_id.values())
    used_playwright = sum(1 for item in items if item.get("method") == "playwright")
    used_alternates = 0
    for result in results_for_items:
        metadata = result.metadata or {}
        source = str(metadata.get("source") or "").lower()
        via = str(metadata.get("via") or "").lower()
        if source in {"override", "brave_search", "perplexity", "alternate", "brave"}:
            used_alternates += 1
            continue
        if via in {"wayback", "jina"}:
            used_alternates += 1
            continue
        if result.method in {"wayback", "jina"}:
            used_alternates += 1
            continue
    downloaded = sum(1 for item in items if (item.get("artifacts") or {}).get("download_path"))
    ok_count = sum(1 for item in items if item.get("verdict") == "ok")
    failed_count = sum(1 for item in items if item.get("verdict") == "failed")

    summary = {
        "command": command,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "started_at": started_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "finished_at": finished_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "duration_ms": int((finished_at - started_at).total_seconds() * 1000),
        "counts": {
            "total": len(items),
            "downloaded": downloaded,
            "ok": ok_count,
            "failed": failed_count,
            "used_playwright": used_playwright,
            "used_alternates": used_alternates,
        },
        "items": items,
    }
    return summary


def _render_walkthrough(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Walkthrough")
    lines.append("")
    lines.append(f"Run ID: {summary.get('run_id')}")
    lines.append(f"Started: {summary.get('started_at')}")
    lines.append(f"Finished: {summary.get('finished_at')}")
    lines.append(f"Duration: {summary.get('duration_ms')} ms")
    lines.append("")
    env_warnings = summary.get("environment_warnings") or []
    if env_warnings:
        lines.append("## Environment Warnings")
        for warning in env_warnings:
            code = warning.get("code", "warning")
            message = warning.get("message", "")
            remedy = warning.get("remedy", "")
            lines.append(f"- {code}: {message}")
            if remedy:
                lines.append(f"  remedy: {remedy}")
        lines.append("")
    soft_failures = summary.get("soft_failures") or []
    if soft_failures:
        lines.append("## Soft Failures")
        for item in soft_failures:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Counts")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    counts = summary.get("counts") or {}
    for key in ("total", "downloaded", "ok", "failed", "used_playwright", "used_alternates"):
        lines.append(f"| {key} | {counts.get(key, 0)} |")
    lines.append("")

    for idx, item in enumerate(summary.get("items") or [], start=1):
        lines.append(f"## Item {idx}")
        lines.append(f"original_url: {item.get('original_url')}")
        lines.append(f"requested_url: {item.get('requested_url')}")
        lines.append(f"final_downloaded_url: {item.get('final_downloaded_url')}")
        lines.append(f"status: {item.get('status')}")
        lines.append(f"verdict: {item.get('verdict')}")
        lines.append(f"paywall_verdict: {item.get('paywall_verdict')}")
        artifacts = item.get("artifacts") or {}
        lines.append(f"download_path: {artifacts.get('download_path')}")
        lines.append(f"extracted_text_path: {artifacts.get('extracted_text_path')}")
        lines.append(f"markdown_path: {artifacts.get('markdown_path')}")
        lines.append(f"fit_markdown_path: {artifacts.get('fit_markdown_path')}")
        warnings = item.get("warnings") or []
        errors = item.get("errors") or []
        lines.append(f"warnings: {', '.join(warnings) if warnings else 'none'}")
        lines.append(f"errors: {', '.join(errors) if errors else 'none'}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_consumer(
    urls: Sequence[str],
    *,
    command: str,
    out_dir: Optional[Path],
    emit: Set[str],
    soft_fail: bool,
) -> Tuple[Dict[str, Any], int]:
    started_at = datetime.now(timezone.utc)
    run_dir, run_id = resolve_run_dir(out_dir)
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to create run dir {run_dir}: {exc}") from exc

    entries = _build_entries(urls)

    env_warnings = collect_environment_warnings()
    soft_failures = [warning.get("code") for warning in env_warnings if warning.get("code")]
    for warning in env_warnings:
        message = warning.get("message") or warning.get("code") or "environment warning"
        remedy = warning.get("remedy")
        if remedy:
            print(f"[fetcher] warning: {message} ({remedy})", file=sys.stderr)
        else:
            print(f"[fetcher] warning: {message}", file=sys.stderr)

    config = FetchConfig()
    config.enable_toc_fanout = False
    config.screenshots_dir = run_dir / "screenshots"
    config.disable_http_cache = os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1"
    if config.overrides_path is None:
        env_overrides = os.getenv("FETCHER_OVERRIDES_PATH")
        config.overrides_path = Path(env_overrides) if env_overrides else DEFAULT_POLICY.overrides_path

    pdf_env = os.getenv("FETCHER_ENABLE_PDF_DISCOVERY")
    if pdf_env:
        config.enable_pdf_discovery = _env_bool("FETCHER_ENABLE_PDF_DISCOVERY", "0")
    pdf_max_env = os.getenv("FETCHER_PDF_DISCOVERY_MAX")
    if pdf_max_env:
        config.pdf_discovery_max = max(0, _env_int("FETCHER_PDF_DISCOVERY_MAX", config.pdf_discovery_max))

    cache_path = resolve_cache_path()
    if config.disable_http_cache:
        cache_path = None
    else:
        config.disable_http_cache = cache_path is None

    fetcher = URLFetcher(config, cache_path=cache_path)
    results, _audit = _run_in_fetch_loop(fetcher.fetch_many(entries))

    emit_fit_md = "fit_md" in emit and "md" in emit
    fit_md_requested = "fit_md" in emit
    download_errors: Optional[Dict[str, Optional[str]]] = None
    if "download" in emit:
        download_errors = persist_downloads(results, run_dir, allow_junk=True)

    for result in results:
        evaluate_result_content(result)
    annotate_paywall_metadata(results, DEFAULT_POLICY)

    if "text" in emit:
        materialize_extracted_text(
            results,
            run_dir,
            enabled=True,
            min_chars=int(getattr(DEFAULT_POLICY, "extracted_text_min_chars", 1) or 1),
        )
    if "md" in emit:
        materialize_markdown(
            results,
            run_dir,
            enabled=True,
            min_chars=int(getattr(DEFAULT_POLICY, "markdown_min_chars", 1) or 1),
            emit_fit_markdown=emit_fit_md,
            fit_min_chars=int(getattr(DEFAULT_POLICY, "fit_markdown_min_chars", 200) or 200),
            overrides_path=getattr(DEFAULT_POLICY, "overrides_path", None),
        )

    finished_at = datetime.now(timezone.utc)
    summary = _build_consumer_summary(
        command=command,
        run_id=run_id,
        run_dir=run_dir,
        started_at=started_at,
        finished_at=finished_at,
        entries=entries,
        results=results,
        emit=emit,
        emit_fit_md=emit_fit_md,
        fit_md_requested=fit_md_requested,
        download_errors=download_errors,
    )
    if env_warnings:
        summary["environment_warnings"] = env_warnings
    if soft_failures:
        summary["soft_failures"] = soft_failures

    summary_path = run_dir / "consumer_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    walkthrough_path = run_dir / "Walkthrough.md"
    walkthrough_path.write_text(_render_walkthrough(summary), encoding="utf-8")

    exit_code = 0
    if not soft_fail:
        if summary["counts"]["failed"] > 0 or soft_failures:
            exit_code = 3

    return summary, exit_code
