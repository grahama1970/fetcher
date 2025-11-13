"""Utilities for paywall detection and alternate URL resolution."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from urllib.parse import urlparse

import requests

from .fetcher_config import BRAVE_ENDPOINT, HDR_ACCEPT, HDR_BRAVE_TOKEN
from .fetcher_utils import normalize_domain as _normalize_domain
from .web_fetch import FetchConfig, FetchResult, URLFetcher
from ..core.keys import (
    K_CONTROLS,
    K_DOMAIN,
    K_FRAMEWORKS,
    K_ORIGINAL_URL,
    K_SOURCE,
    K_TITLE,
    K_TITLES,
    K_URL,
    K_WORKSHEETS,
)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .fetcher import FetcherPolicy


DEFAULT_OVERRIDE_RULES: List[Dict[str, Any]] = [
    {"domain": "twitter.com", "target_template": "https://r.jina.ai/{original}"},
    {"domain": "x.com", "target_template": "https://r.jina.ai/{original}"},
    {
        "domain": "atlas.mitre.org",
        "target_url": "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/ATLAS.yaml",
    },
    {"substring": "jmichaelwaller.com", "target_local_file": "IM-Telstar-Aug2003.txt"},
    {
        "domain": "spoonfeedin.blogspot.com.au",
        "target_url": "https://www.ndss-symposium.org/wp-content/uploads/spacesec2024-87-paper.pdf",
    },
    {
        "domain": "money.cnn.com",
        "target_url": "https://www.usatoday.com/story/weather/2014/11/12/china-weather-satellite-attack/18915137/",
    },
    {
        "domain": "securityboulevard.com",
        "target_url": "https://www.splunk.com/en_us/blog/learn/watering-hole-attacks.html",
    },
    {
        "substring": "news24.com",
        "target_url": "https://www.voanews.com/a/a-13-a-2002-07-08-13-china-67573262/388684.html",
    },
    {
        "domain": "cheatsheetseries.owasp.org",
        "path_prefix": "/cheatsheets/Abuse_Case_Cheat_She",
        "target_url": "https://cheatsheetseries.owasp.org/cheatsheets/Abuse_Case_Cheat_Sheet.html",
    },
]

_OVERRIDE_RULES: Optional[List[Dict[str, Any]]] = None
_PAYWALL_VERDICT_ALLOW = frozenset({"maybe", "likely"})


def reload_overrides_cache() -> None:
    global _OVERRIDE_RULES
    _OVERRIDE_RULES = None


def _load_override_rules(overrides_path: Path, allow_reload: bool) -> List[Dict[str, Any]]:
    global _OVERRIDE_RULES
    if _OVERRIDE_RULES is not None and not allow_reload:
        return _OVERRIDE_RULES
    rules: List[Dict[str, Any]] = []
    try:
        if overrides_path.exists():
            data = json.loads(overrides_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rules = data
    except Exception:
        rules = []
    _OVERRIDE_RULES = rules
    return rules


def _merge_paywall_hints(policy: "FetcherPolicy") -> Tuple[str, ...]:
    if not policy.paywall_hints_path:
        return policy.paywall_hints
    try:
        path = policy.paywall_hints_path
        if path and path.exists():
            text = path.read_text(encoding="utf-8")
            lines: List[str] = []
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                lines.append(line)
            return tuple({*policy.paywall_hints, *lines})
    except Exception:
        pass
    return policy.paywall_hints


def _verdict_allows_resolver(result: Optional[FetchResult]) -> bool:
    """Return True when verdict indicates we should keep resolving."""

    if result is None:
        return True
    metadata = result.metadata or {}
    verdict = metadata.get("paywall_verdict")
    if verdict is None:
        detection = metadata.get("paywall_detection") or {}
        verdict = detection.get("verdict")
    if verdict is None:
        return True
    return verdict in _PAYWALL_VERDICT_ALLOW


def looks_paywalled_url(url: str, summary: Optional[str], policy: "FetcherPolicy") -> bool:
    lower_url = url.lower()
    if any(token in lower_url for token in ("login", "signin", "subscribe", "paywall", "account")):
        return True
    if summary:
        lowered = summary.lower()
        hints = _merge_paywall_hints(policy)
        if any(hint in lowered for hint in hints):
            return True
    return False


def load_control_contexts(root: Path) -> Dict[str, Dict[str, Any]]:
    contexts_path = root / "data" / "processed" / "controls_context.jsonl"
    contexts: Dict[str, Dict[str, Any]] = {}
    if not contexts_path.exists():
        return contexts
    with contexts_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            control_id = data.get("id")
            if control_id:
                contexts[control_id] = data
    return contexts


def _validate_candidate_url(
    url: str,
    config: FetchConfig,
    policy: "FetcherPolicy",
    run_fetch_loop: Callable[[Any], Any],
) -> Optional[FetchResult]:
    entry = {
        K_URL: url,
        K_DOMAIN: urlparse(url).netloc,
        K_CONTROLS: [],
        K_TITLES: [],
        K_FRAMEWORKS: [],
        K_WORKSHEETS: [],
    }
    v = policy.validator
    validator_config = FetchConfig(
        concurrency=v.concurrency,
        per_domain=v.per_domain,
        timeout=(v.timeout if v.timeout is not None else config.timeout),
        max_attempts=v.max_attempts,
        local_data_root=config.local_data_root,
        insecure_ssl_domains=config.insecure_ssl_domains,
    )
    validator = URLFetcher(validator_config)
    results, _ = run_fetch_loop(validator.fetch_many([entry]))
    if not results:
        return None
    result = results[0]
    if result.status == 200 and result.text:
        return result
    return None


def _persist_alternate_updates(
    inventory_path: Path,
    run_artifacts_dir: Path,
    url_updates: Dict[str, str],
    applied: List[Dict[str, Any]],
) -> None:
    if not applied:
        return
    provider_map = {record["old_url"]: record.get("provider", "alternate") for record in applied}
    lines = inventory_path.read_text(encoding="utf-8").splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            new_lines.append(line)
            continue
        entry = json.loads(line)
        url = entry.get(K_URL)
        original = entry.get(K_ORIGINAL_URL) or url
        if original in url_updates:
            new_url = url_updates[original]
            entry.setdefault(K_ORIGINAL_URL, original)
            entry[K_URL] = new_url
            entry[K_DOMAIN] = urlparse(new_url).netloc
            entry[K_SOURCE] = provider_map.get(original, entry.get(K_SOURCE) or "alternate")
        new_lines.append(json.dumps(entry, ensure_ascii=False))
    inventory_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    log_path = run_artifacts_dir / "alternate_urls_applied.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        for record in applied:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _stable_rule_id(rule: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(rule, sort_keys=True, ensure_ascii=True)
    except Exception:
        payload = str(rule)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _direct_override_candidate(entry: Dict[str, Any], policy: "FetcherPolicy") -> Optional[Tuple[str, Dict[str, Any]]]:
    url = entry.get(K_URL, "").strip()
    if not url:
        return None

    parsed = urlparse(url)
    original = (entry.get(K_ORIGINAL_URL) or url).strip()
    raw_domain = (entry.get(K_DOMAIN) or parsed.netloc or urlparse(original).netloc)
    domain = _normalize_domain(raw_domain, set(policy.strip_subdomains))

    try:
        rules = list(_load_override_rules(policy.overrides_path, policy.overrides_reload)) + list(DEFAULT_OVERRIDE_RULES)
        base_url = original if original.startswith("http") else url
        if not (base_url or "").startswith("http"):
            scheme = parsed.scheme or "https"
            netloc = parsed.netloc or raw_domain
            path = parsed.path or ""
            query = f"?{parsed.query}" if parsed.query else ""
            base_url = f"{scheme}://{netloc}{path}{query}"
        for rule in rules:
            rd = _normalize_domain(str(rule.get("domain") or "").strip(), set(policy.strip_subdomains))
            if rd and rd != domain:
                continue
            substr = str(rule.get("substring") or "").strip()
            if substr and substr not in (original or url):
                continue
            prefix = str(rule.get("path_prefix") or "").strip()
            if prefix and not (parsed.path or "").startswith(prefix):
                continue
            qmatch = rule.get("query_contains")
            if isinstance(qmatch, dict) and qmatch:
                q = parsed.query or ""
                if not all((f"{k}={v}" in q) for k, v in qmatch.items()):
                    continue
            pregex = rule.get("path_regex")
            if isinstance(pregex, str) and pregex:
                try:
                    if not re.search(pregex, parsed.path or ""):
                        continue
                except re.error:
                    continue
            tmpl = rule.get("target_template")
            if isinstance(tmpl, str) and tmpl:
                return tmpl.replace("{original}", base_url), rule
            target = rule.get("target_url")
            if isinstance(target, str) and target:
                return target, rule
            local_file = rule.get("target_local_file")
            if isinstance(local_file, str) and local_file:
                local_path = (policy.local_sources_dir / local_file).resolve()
                if local_path.exists():
                    return local_path.as_uri(), rule
    except Exception:
        pass

    return None


def _search_brave_for_alternates(
    targets: List[Tuple[str, Tuple[int, Dict[str, Any]]]],
    control_contexts: Dict[str, Dict[str, Any]],
    config: FetchConfig,
    policy: "FetcherPolicy",
    run_fetch_loop: Callable[[Any], Any],
    debug_log_path: Optional[Path] = None,
) -> Dict[str, Dict[str, Any]]:
    api_key = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")
    if not api_key:
        return {}

    endpoint = BRAVE_ENDPOINT
    headers = {HDR_ACCEPT: "application/json", HDR_BRAVE_TOKEN: api_key}

    applied: Dict[str, Dict[str, Any]] = {}
    for original_url, (_idx, entry) in targets:
        control_id = (entry.get(K_CONTROLS) or [original_url])[0]
        context = control_contexts.get(control_id, {})
        query_parts = [context.get("title") or entry.get(K_TITLES, [""])[0]]
        definition = context.get("definition") or ""
        if definition:
            query_parts.append(definition[:160])
        query_parts.append(entry.get(K_DOMAIN) or "")
        query = " ".join(part for part in query_parts if part).strip()
        if not query:
            query = original_url

        try:
            resp = requests.get(
                endpoint,
                headers=headers,
                params={"q": query, "count": 10},
                timeout=15,
            )
            if resp.status_code != 200:
                continue
            payload = resp.json()
        except Exception:
            continue

        web_results = payload.get("web", {}).get("results", []) if isinstance(payload, dict) else []
        if debug_log_path and policy.debug.brave_top_n > 0:
            try:
                debug_log_path.parent.mkdir(parents=True, exist_ok=True)
                with debug_log_path.open("a", encoding="utf-8") as fh:
                    for item in web_results[: policy.debug.brave_top_n]:
                        fh.write(
                            json.dumps(
                                {
                                    "original_url": original_url,
                                    "candidate_url": (item.get("url") or item.get("link") or "").strip(),
                                    "title": item.get("title"),
                                    "description": item.get("description"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            except Exception:
                pass
        for item in web_results:
            candidate_url = (item.get("url") or item.get("link") or "").strip()
            if not candidate_url:
                continue
            domain = _normalize_domain(urlparse(candidate_url).netloc, set(policy.strip_subdomains))
            if domain in policy.brave_excluded_domains:
                continue
            summary = item.get("description") or item.get("title") or ""
            if looks_paywalled_url(candidate_url, summary, policy):
                continue
            validation = _validate_candidate_url(candidate_url, config, policy, run_fetch_loop)
            if not validation:
                continue
            applied[original_url] = {
                "url": candidate_url,
                "summary": summary,
                "title": item.get("title"),
                "validation": validation,
            }
            break
    return applied


def resolve_paywalled_entries(
    entries: List[Dict[str, Any]],
    results: List[FetchResult],
    config: FetchConfig,
    root: Path,
    inventory_path: Path,
    run_artifacts_dir: Path,
    limit: int,
    policy: "FetcherPolicy",
    run_fetch_loop: Callable[[Any], Any],
) -> List[Dict[str, Any]]:
    try:
        from .alternate_urls import FailedControl, generate_alternate_urls
    except Exception:
        return []

    results_map = {result.url: result for result in results}
    applied: List[Dict[str, Any]] = []
    url_updates: Dict[str, str] = {}
    control_contexts = load_control_contexts(root)
    pending: Dict[str, Tuple[int, Dict[str, Any]]] = {}

    for idx, entry in enumerate(entries):
        url = entry.get(K_URL, "").strip()
        if not url:
            continue

        result = results_map.get(url)
        status = result.status if result else -1
        raw_domain = (entry.get(K_DOMAIN) or urlparse(url).netloc)
        domain = _normalize_domain(raw_domain, set(policy.strip_subdomains))

        original = entry.get(K_ORIGINAL_URL, url)

        override_hit = _direct_override_candidate(entry, policy)
        if override_hit:
            override_candidate, rule = override_hit
            validation = _validate_candidate_url(override_candidate, config, policy, run_fetch_loop)
            if validation:
                current_url = entry.get(K_URL, original)
                entry.setdefault(K_ORIGINAL_URL, original)
                entry[K_URL] = override_candidate
                entry[K_DOMAIN] = urlparse(override_candidate).netloc
                entry[K_SOURCE] = entry.get(K_SOURCE) or "override"

                results[:] = [r for r in results if r.url != current_url]
                results.append(validation)

                url_updates[original] = override_candidate
                record = {
                    "control_id": (entry.get(K_CONTROLS) or [original])[0],
                    "old_url": original,
                    "new_url": override_candidate,
                    "provider": "override",
                    "summary": "domain-specific override",
                }
                if policy.debug.log_rule_hits:
                    record["rule_id"] = _stable_rule_id(rule)
                    record["rule_fields"] = {
                        k: v
                        for k, v in rule.items()
                        if k in {"domain", "substring", "path_prefix", "target_url", "target_template", "target_local_file"}
                    }
                applied.append(record)
                continue

        is_404 = (status == 404)
        if (status not in policy.paywall_status_codes) and not is_404:
            continue
        if domain not in policy.paywall_domains:
            continue
        if not _verdict_allows_resolver(result):
            continue

        if original in pending:
            continue
        pending[original] = (idx, entry)

    if not pending:
        if applied:
            _persist_alternate_updates(inventory_path, run_artifacts_dir, url_updates, applied)
        return applied

    targets = list(pending.items())[:limit]
    brave_debug_path = (run_artifacts_dir / "brave_candidates_debug.jsonl") if policy.debug.brave_top_n > 0 else None
    brave_matches = _search_brave_for_alternates(targets, control_contexts, config, policy, run_fetch_loop, brave_debug_path)

    for original_url, data in brave_matches.items():
        if original_url not in pending:
            continue
        idx, entry = pending.pop(original_url)
        current_url = entry.get(K_URL, original_url)
        entry.setdefault(K_ORIGINAL_URL, original_url)
        entry[K_URL] = data["url"]
        entry[K_DOMAIN] = urlparse(data["url"]).netloc
        entry[K_SOURCE] = "brave_search"

        results[:] = [r for r in results if r.url != current_url]
        results.append(data["validation"])

        url_updates[original_url] = data["url"]
        applied.append(
            {
                "control_id": (entry.get(K_CONTROLS) or [original_url])[0],
                "old_url": original_url,
                "new_url": data["url"],
                "provider": "brave",
                "summary": data.get("summary"),
            }
        )

    if not pending:
        if applied:
            _persist_alternate_updates(inventory_path, run_artifacts_dir, url_updates, applied)
        return applied

    if not (policy.enable_perplexity_resolver and os.getenv("PERPLEXITY_API_KEY")):
        if applied:
            _persist_alternate_updates(inventory_path, run_artifacts_dir, url_updates, applied)
        return applied

    remaining_targets = list(pending.items())[: max(0, limit - len(applied))]
    controls: List[Any] = []

    for original_url, (_idx, entry) in remaining_targets:
        controls_list = entry.get(K_CONTROLS) or []
        titles = entry.get(K_TITLES) or []
        worksheets = entry.get(K_WORKSHEETS) or []
        control_id = controls_list[0] if controls_list else original_url
        title = titles[0] if titles else entry.get(K_TITLE) or ""
        context = control_contexts.get(control_id, {})
        controls.append(
            FailedControl(
                control_id=control_id,
                title=title,
                failed_url=original_url,
                worksheet=worksheets[0] if worksheets else None,
                domain=entry.get(K_DOMAIN),
                category=context.get("category"),
                definition=context.get("definition"),
                source_title=context.get("title") or title,
            )
        )

    debug_path = run_artifacts_dir / "alternate_urls_debug.jsonl"
    debug_records: List[Dict[str, Any]] = []

    successes, failures = generate_alternate_urls(controls, concurrency=3)

    for result in successes:
        debug_records.append(
            {
                "control_id": result.control.control_id,
                "failed_url": result.control.failed_url,
                "suggestion_count": len(result.suggestions),
                "raw_response_excerpt": result.raw_response[:240],
            }
        )
    for result in failures:
        debug_records.append(
            {
                "control_id": result.control.control_id,
                "failed_url": result.control.failed_url,
                "error": result.error,
                "raw_response_excerpt": result.raw_response[:240],
            }
        )

    if debug_records:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("a", encoding="utf-8") as fh:
            for record in debug_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    for result in successes:
        if not result.suggestions:
            continue
        suggestion = result.suggestions[0]
        candidate_url = suggestion.suggested_url.strip()
        if not candidate_url:
            continue
        validation = _validate_candidate_url(candidate_url, config, policy, run_fetch_loop)
        if not validation:
            continue

        original_url = result.control.failed_url
        if original_url not in pending:
            continue

        _idx, entry = pending.pop(original_url)
        current_url = entry.get(K_URL, original_url)

        entry.setdefault(K_ORIGINAL_URL, original_url)
        entry[K_URL] = candidate_url
        entry[K_DOMAIN] = urlparse(candidate_url).netloc
        entry[K_SOURCE] = suggestion.provider or "perplexity"

        results[:] = [r for r in results if r.url != current_url]
        results.append(validation)

        url_updates[original_url] = candidate_url
        applied.append(
            {
                "control_id": result.control.control_id,
                "old_url": original_url,
                "new_url": candidate_url,
                "provider": suggestion.provider or "perplexity",
                "summary": suggestion.summary,
            }
        )

    if applied:
        _persist_alternate_updates(inventory_path, run_artifacts_dir, url_updates, applied)

    return applied


def sanity_check() -> None:
    class _Dummy:
        def __init__(self, verdict: Optional[str]):
            self.status = 200
            self.metadata = {"paywall_verdict": verdict} if verdict else {}

    assert not _verdict_allows_resolver(_Dummy("unlikely"))
    assert _verdict_allows_resolver(_Dummy("maybe"))
    assert _verdict_allows_resolver(None)


sanity_check()


__all__ = [
    "DEFAULT_OVERRIDE_RULES",
    "reload_overrides_cache",
    "looks_paywalled_url",
    "load_control_contexts",
    "resolve_paywalled_entries",
    "sanity_check",
]
