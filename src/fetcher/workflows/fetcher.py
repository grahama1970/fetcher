"""Reusable helpers for Sparta URL fetching and alternate resolution."""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
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

# ---------------------------------------------------------------------------
# Constants shared with the pipeline entrypoint.
# ---------------------------------------------------------------------------

def _idna_normalize(host: str) -> str:
    h = (host or "").strip().rstrip(".").lower()
    if not h:
        return ""
    try:
        h = h.encode("idna").decode("ascii")
    except Exception:
        pass
    return h


def _normalize_domain(domain: str, strip_subdomains: Set[str]) -> str:
    d = _idna_normalize(domain)
    if not d:
        return d
    labels = d.split(".")
    if len(labels) > 2 and labels[0] in strip_subdomains:
        return ".".join(labels[1:])
    return d


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


def _normalize_set(domains: Iterable[str], strip: Set[str]) -> Set[str]:
    return {_normalize_domain(d, strip) for d in domains}


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

# Built-in, data-driven overrides replacing scattered string branches.
# These can be extended in data/overrides.json without code changes.
DEFAULT_OVERRIDE_RULES: List[Dict[str, Any]] = [
    # Template rehost for social/X
    {"domain": "twitter.com", "target_template": "https://r.jina.ai/{original}"},
    {"domain": "x.com", "target_template": "https://r.jina.ai/{original}"},
    # MITRE ATLAS canonical raw
    {
        "domain": "atlas.mitre.org",
        "target_url": "https://raw.githubusercontent.com/mitre-atlas/atlas-data/main/dist/ATLAS.yaml",
    },
    # Local mirror file when present
    {"substring": "jmichaelwaller.com", "target_local_file": "IM-Telstar-Aug2003.txt"},
    # Specific alternates previously hardcoded
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
    # OWASP cheat sheet truncated path fix
    {
        "domain": "cheatsheetseries.owasp.org",
        "path_prefix": "/cheatsheets/Abuse_Case_Cheat_She",
        "target_url": "https://cheatsheetseries.owasp.org/cheatsheets/Abuse_Case_Cheat_Sheet.html",
    },
]

_OVERRIDE_RULES: Optional[List[Dict[str, Any]]] = None


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


@dataclass(slots=True)
class FetcherResult:
    """Outcome payload returned to pipeline steps."""

    results: List[FetchResult]
    audit: Dict[str, Any]
    applied_alternates: List[Dict[str, Any]]
    outstanding_controls: int
    outstanding_urls: int


def _merge_paywall_hints(policy: FetcherPolicy) -> Tuple[str, ...]:
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


def _looks_paywalled_url(url: str, summary: Optional[str], policy: FetcherPolicy = DEFAULT_POLICY) -> bool:
    lower_url = url.lower()
    if any(token in lower_url for token in ("login", "signin", "subscribe", "paywall", "account")):
        return True
    if summary:
        lowered = summary.lower()
        hints = _merge_paywall_hints(policy)
        if any(hint in lowered for hint in hints):
            return True
    return False


def _load_control_contexts(root: Path) -> Dict[str, Dict[str, Any]]:
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


def _resolve_repo_root(inventory_path: Path) -> Path:
    """Best-effort discovery of the repository root for data assets.

    The original implementation assumed ``inventory_path`` lived three levels
    below the repo root. That fails for bespoke pipelines that stage the
    inventory in a temporary directory. We instead climb ancestors until we
    find a ``data/processed`` directory, falling back to the immediate parent
    when nothing matches.
    """

    search: List[Path] = [inventory_path.parent]
    search.extend(list(inventory_path.parents))
    seen: Set[Path] = set()
    for candidate in search:
        if candidate in seen:
            continue
        seen.add(candidate)
        data_dir = candidate / "data" / "processed"
        if data_dir.exists():
            return candidate
    return inventory_path.parent


def _validate_candidate_url(url: str, config: FetchConfig, policy: FetcherPolicy) -> Optional[FetchResult]:
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
    results, _ = _run_in_fetch_loop(validator.fetch_many([entry]))
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


def _direct_override_candidate(entry: Dict[str, Any], policy: FetcherPolicy = DEFAULT_POLICY) -> Optional[Tuple[str, Dict[str, Any]]]:
    url = entry.get(K_URL, "").strip()
    if not url:
        return None

    parsed = urlparse(url)
    original = (entry.get(K_ORIGINAL_URL) or url).strip()
    raw_domain = (entry.get(K_DOMAIN) or parsed.netloc or urlparse(original).netloc)
    domain = _normalize_domain(raw_domain, set(policy.strip_subdomains))

    # Data-driven overrides (file + built-ins); avoids hardcoded branches.
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
            # Optional query_contains: {k:v} substring check
            qmatch = rule.get("query_contains")
            if isinstance(qmatch, dict) and qmatch:
                q = parsed.query or ""
                if not all((f"{k}={v}" in q) for k, v in qmatch.items()):
                    continue
            # Optional path_regex selector
            pregex = rule.get("path_regex")
            if isinstance(pregex, str) and pregex:
                try:
                    if not re.search(pregex, parsed.path or ""):
                        continue
                except re.error:
                    continue
            # Support template
            tmpl = rule.get("target_template")
            if isinstance(tmpl, str) and tmpl:
                return tmpl.replace("{original}", base_url), rule
            # Support explicit URL
            target = rule.get("target_url")
            if isinstance(target, str) and target:
                return target, rule
            # Support local file resolution
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
    policy: FetcherPolicy,
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
        # Optional debug log of top-N candidates
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
            if _looks_paywalled_url(candidate_url, summary, policy):
                continue
            validation = _validate_candidate_url(candidate_url, config, policy)
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


def _resolve_paywalled_entries(
    entries: List[Dict[str, Any]],
    results: List[FetchResult],
    config: FetchConfig,
    root: Path,
    inventory_path: Path,
    run_artifacts_dir: Path,
    limit: int,
    policy: FetcherPolicy,
) -> List[Dict[str, Any]]:
    try:
        from .alternate_urls import FailedControl, generate_alternate_urls
    except Exception:
        return []

    results_map = {result.url: result for result in results}
    applied: List[Dict[str, Any]] = []
    url_updates: Dict[str, str] = {}
    control_contexts = _load_control_contexts(root)
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

        # 1) Try data-driven overrides FIRST (regardless of domain allowlist).
        # This ensures explicit rules in overrides.json are honored even when
        # a domain isn't flagged as paywalled.
        override_hit = _direct_override_candidate(entry, policy)
        if override_hit:
            override_candidate, rule = override_hit
            validation = _validate_candidate_url(override_candidate, config, policy)
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
                        if k
                        in (
                            "domain",
                            "substring",
                            "path_prefix",
                            "target_url",
                            "target_template",
                            "target_local_file",
                        )
                    }
                applied.append(record)
                continue

        # 2) If no explicit override, apply paywall/alternate logic.
        is_404 = (status == 404)
        if (status not in policy.paywall_status_codes) and not is_404:
            continue
        if domain not in policy.paywall_domains:
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
    brave_matches = _search_brave_for_alternates(targets, control_contexts, config, policy, brave_debug_path)

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
    controls: List[FailedControl] = []
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
        validation = _validate_candidate_url(candidate_url, config, policy)
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


def _maybe_externalize_text(
    results: Iterable[FetchResult],
    run_artifacts_dir: Path,
    max_inline_bytes: int,
) -> None:
    """Persist large response bodies to disk to keep JSONL payloads small."""

    if max_inline_bytes <= 0:
        return

    text_root = run_artifacts_dir / "text_blobs"
    text_root.mkdir(parents=True, exist_ok=True)

    for result in results:
        text = result.text or ""
        if not text:
            continue
        text_bytes = text.encode("utf-8")
        if len(text_bytes) <= max_inline_bytes:
            continue

        sha = hashlib.sha256(text_bytes).hexdigest()
        blob_path = text_root / f"{sha}.txt"
        blob_path.write_text(text, encoding="utf-8")

        metadata = dict(result.metadata or {})
        metadata[K_TEXT_PATH] = str(blob_path)
        metadata["text_externalized"] = True
        metadata["text_inline_bytes"] = len(text_bytes)
        metadata["text_sha256"] = sha

        result.metadata = metadata
        result.text = ""


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

    repo_root = _resolve_repo_root(inventory_path)

    applied: List[Dict[str, Any]] = []
    if enable_resolver and entries:
        applied = _resolve_paywalled_entries(
            entries,
            results,
            fetch_config,
            repo_root,
            inventory_path,
            run_artifacts_dir,
            limit=resolver_limit,
            policy=policy,
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

    _maybe_externalize_text(results, run_artifacts_dir, policy.text_inline_max_bytes)

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
    return results[0]


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
