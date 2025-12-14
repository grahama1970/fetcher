"""Helper utilities for building outstanding URL reports."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

from ..core.keys import (
    K_CONTROLS,
    K_DOMAIN,
    K_ORIGINAL_URL,
    K_STATUS,
    K_TITLES,
    K_URL,
    K_WORKSHEETS,
)
from .web_fetch import FetchResult, SPA_FALLBACK_DOMAINS, SIGNIN_MODAL_DOMAINS
from . import fetcher_utils

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .fetcher import FetcherPolicy


def build_outstanding_reports(
    entries: Iterable[dict],
    results: List[FetchResult],
    run_artifacts_dir: Path,
    policy: "FetcherPolicy",
) -> Tuple[int, int]:
    results_map: Dict[str, FetchResult] = {}
    for result in results:
        results_map[result.url] = result
        original = (result.metadata or {}).get("original_url")
        if original:
            results_map.setdefault(original, result)

    outstanding_controls: List[Dict[str, Any]] = []
    outstanding_urls: Dict[str, Dict[str, Any]] = {}

    for entry in entries:
        url = entry[K_URL].strip()
        result = results_map.get(url)
        status = result.status if result else -1
        text_ok = bool(result and result.status == 200 and fetcher_utils.has_text_payload(result))
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

    url_rollup: List[Dict[str, Any]] = []
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

    summary = generate_outstanding_summary(url_rollup, results_map, policy)
    (run_artifacts_dir / "outstanding_domains_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return len(outstanding_controls), len(url_rollup)


def generate_outstanding_summary(
    url_rollup: List[Dict[str, Any]],
    results_map: Dict[str, FetchResult],
    policy: "FetcherPolicy",
) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    generated_at = datetime.now(timezone.utc).isoformat()
    safe_suffixes = getattr(policy, "paywall_safe_suffixes", tuple())
    strip = set(getattr(policy, "strip_subdomains", frozenset({"www"})))

    for entry in url_rollup:
        url = entry.get(K_URL, "")
        domain = entry.get(K_DOMAIN) or urlparse(url).netloc
        raw_status = entry.get(K_STATUS, -1)
        try:
            status = int(raw_status)
        except (TypeError, ValueError):
            status = -1
        result = results_map.get(url)
        metadata = result.metadata if result else {}
        content_verdict = metadata.get("content_verdict")
        verdict = metadata.get("paywall_verdict")
        if verdict is None:
            detection = metadata.get("paywall_detection") or {}
            verdict = detection.get("verdict")
        score = metadata.get("paywall_score")
        safe_domain = fetcher_utils.is_safe_domain(domain, policy.paywall_safe_domains, safe_suffixes, strip)
        category, note = _categorize_outstanding(
            domain,
            status,
            verdict,
            safe_domain,
            policy,
            content_verdict,
            metadata,
        )
        counts[category] = counts.get(category, 0) + 1
        items.append(
            {
                "url": url,
                "domain": domain,
                "status": status,
                "worksheets": entry.get("worksheets", []),
                "control_count": entry.get("control_count", 0),
                "paywall_verdict": verdict,
                "paywall_score": score,
                "safe_domain": safe_domain,
                "brave_suggestion": metadata.get("brave_suggestion"),
                "category": category,
                "notes": note,
                "fetch_method": getattr(result, "method", None) if result else None,
            }
        )

    return {
        "generated_at": generated_at,
        "total": len(items),
        "counts_by_category": counts,
        "items": items,
    }


def _categorize_outstanding(
    domain: str,
    status: int,
    verdict: Optional[str],
    safe_domain: bool,
    policy: "FetcherPolicy",
    content_verdict: Optional[str],
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    metadata = metadata or {}
    lower_verdict = (verdict or "").lower()
    content_verdict = (content_verdict or "").lower()

    if status in {404, 410, 0, -1}:
        return "broken_or_moved", f"HTTP status {status}"

    if status in {500, 502, 503}:
        return "retry", f"HTTP status {status}"

    # Distinguish bot/anti-automation interstitials (e.g., Cloudflare "Just a moment")
    # from generic login/JS-required pages. These typically present short HTML
    # with characteristic titles/text and block both plain HTTP and Playwright.
    if status in {403, 503}:
        hub_title = (metadata.get("hub_title") or "").lower()
        excerpt = (metadata.get("content_excerpt") or "").lower()
        bot_markers = ("just a moment", "cloudflare", "cf-error-details", "cf-challenge")
        if any(token in hub_title for token in ("just a moment",)) or any(
            token in excerpt for token in bot_markers
        ):
            return "bot_blocked", "Automated access blocked by anti-bot interstitial"

    if status in {401, 403} and domain in (SIGNIN_MODAL_DOMAINS | SPA_FALLBACK_DOMAINS):
        return "needs_login_or_playwright", "Requires login or JS rendering"

    if lower_verdict in {"likely", "maybe"}:
        return "paywall", f"Paywall verdict {lower_verdict}"

    if content_verdict == "missing_file":
        return "broken_or_moved", "content_verdict=missing_file"

    if content_verdict in {"paywall", "thin", "link_hub", "weak"}:
        reason = "content_verdict=" + content_verdict
        mapping = {
            "paywall": "paywall",
            "thin": "content_thin",
            "link_hub": "content_link_hub",
            "weak": "content_weak",
        }
        return mapping.get(content_verdict, "content_issue"), reason

    if content_verdict == "password_protected":
        return "password_protected", "content_verdict=password_protected"

    if safe_domain:
        return "needs_whitelist", "Domain matches safe-domain policy"

    if status in policy.paywall_status_codes:
        return "retry", f"HTTP status {status}"

    return "retry", "Fetch failed for unknown reasons"


__all__ = ["build_outstanding_reports", "generate_outstanding_summary"]
