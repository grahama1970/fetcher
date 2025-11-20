from __future__ import annotations

import asyncio
import csv
import hashlib
import re
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import unquote, urlparse, urlunparse

import aiohttp
try:  # Prefer classic fitz alias; fall back to pymupdf if needed
    import fitz  # type: ignore
except ImportError:  # pragma: no cover - environment-specific
    import pymupdf as fitz  # type: ignore
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning  # type: ignore
from urllib.parse import urljoin, quote
import os
import warnings
import requests
from .prefilters import evaluate_body_prefilter
from . import github_utils
from .extract_utils import extract_content_features
from .fetcher_utils import has_text_payload

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

try:  # Optional OCR fallback for image-heavy PDFs
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Image = None  # type: ignore

try:  # Playwright is optional; fallback gracefully if unavailable
    from playwright.async_api import async_playwright  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    async_playwright = None  # type: ignore

SPA_FALLBACK_DOMAINS = {
    "d3fend.mitre.org",
    "atlas.mitre.org",
    "www.reuters.com",
    "www.oig.nasa.gov",
    "policy.defense.gov",
    "www.forbes.com",
    "www.gps.gov",
    "asiatimes.com",
    "www.linkedin.com",
    "linkedin.com",
    "www.cisa.gov",
    "cisa.gov",
    "www.cnas.org",
    "cnas.org",
    "www.cpomagazine.com",
    "cpomagazine.com",
    "www.cyberdefensemagazine.com",
    "cyberdefensemagazine.com",
    "www.darkreading.com",
    "darkreading.com",
    "online.wsj.com",
    "www.mdpi.com",
    "mdpi.com",
    "www.sentinelone.com",
    "sentinelone.com",
    "www.splunk.com",
    "splunk.com",
}

SIGNIN_MODAL_DOMAINS = {
    "www.linkedin.com",
    "linkedin.com",
    "www.cisa.gov",
    "cisa.gov",
    "www.cnas.org",
    "cnas.org",
    "www.cpomagazine.com",
    "cpomagazine.com",
    "www.cyberdefensemagazine.com",
    "cyberdefensemagazine.com",
    "www.darkreading.com",
    "darkreading.com",
    "online.wsj.com",
    "www.mdpi.com",
    "mdpi.com",
    "www.sentinelone.com",
    "sentinelone.com",
    "www.splunk.com",
    "splunk.com",
}

def _normalize_url(u: str) -> str:
    """Normalize a URL for cache/dedup purposes.

    - Strip trailing slash (except root)
    - Lower-case host
    - Remove default ports
    """
    try:
        u = (u or "").strip()
        if len(u) > 1 and u.endswith("/"):
            u = u[:-1]
        p = urlparse(u)
        host = p.hostname.lower() if p.hostname else None
        netloc = host if host else p.netloc
        # Drop default ports
        if p.port and ((p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443)):
            netloc = host or netloc
        p = p._replace(netloc=netloc)
        return urlunparse(p)
    except Exception:
        return u

def _toggle_trailing_slash(u: str) -> str:
    """Return the same URL with trailing slash toggled.

    - If path ends with a slash (and is not root), remove it.
    - Otherwise append a slash to non-empty path.
    """
    try:
        p = urlparse(u)
        path = p.path or ""
        if path.endswith("/") and len(path) > 1:
            path = path.rstrip("/")
        elif not path.endswith("/") and len(path) >= 1:
            path = path + "/"
        else:
            return u
        return urlunparse(p._replace(path=path))
    except Exception:
        return u


@dataclass
class FetchConfig:
    """Configuration parameters for asynchronous URL fetching."""

    concurrency: int = 24
    per_domain: int = 4
    timeout: float = 20.0
    max_attempts: int = 3
    backoff_initial: float = 0.8
    backoff_max: float = 6.0
    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
    accept_language: str = "en-US,en;q=0.9"
    js_text_threshold: int = 1024
    min_text_length: int = 120
    verify_html: bool = True
    insecure_ssl_domains: Tuple[str, ...] = ()
    local_data_root: Optional[Path] = None
    screenshots_dir: Optional[Path] = None


@dataclass
class FetchResult:
    """Container for a single fetch attempt."""

    url: str
    domain: str
    status: int
    content_type: str
    text: str
    fetched_at: str
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    from_cache: bool = False
    error: Optional[str] = None
    raw_bytes: Optional[bytes] = field(default=None, repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "url": self.url,
            "domain": self.domain,
            "status": self.status,
            "content_type": self.content_type,
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            if self.text
            else "",
            "text_length": len(self.text),
            "fetched_at": self.fetched_at,
            "method": self.method,
            "from_cache": self.from_cache,
        }
        if self.text:
            payload["text"] = self.text
        if self.error:
            payload["error"] = self.error
        if self.metadata:
            payload.update(self.metadata)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class URLFetcher:
    """Async URL fetcher with per-domain throttling and Playwright fallback."""

    def __init__(
        self,
        config: FetchConfig,
        cache_path: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        if cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}
        self._cache_dirty = False
        # Guard cache mutations across async tasks sharing the same event loop
        self._cache_lock: asyncio.Lock = asyncio.Lock()
        self._local_data_root = (
            Path(config.local_data_root).resolve()
            if config.local_data_root is not None
            else None
        )
        self._d3fend_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._insecure_ssl_domains = {domain.lower() for domain in config.insecure_ssl_domains}

    async def fetch_many(
        self,
        entries: Iterable[Dict[str, Any]],
        progress_hook: Optional[Callable[[int, int, Dict[str, Any], FetchResult], None]] = None,
    ) -> Tuple[List[FetchResult], Dict[str, Any]]:
        # Deduplicate by normalized URL so variants (trailing slash, default ports) do not refetch
        # Merge metadata across duplicates (union of controls/titles/frameworks/worksheets/instances)
        dedup: Dict[str, Dict[str, Any]] = {}
        for entry in entries:
            raw = (entry.get("url") or "").strip()
            norm = _normalize_url(raw)
            new_entry = {**entry}
            new_entry["url_normalized"] = norm
            if norm not in dedup:
                dedup[norm] = new_entry
            else:
                cur = dedup[norm]
                # merge list-like fields by union
                for key in ("controls", "titles", "frameworks", "worksheets"):
                    a = cur.get(key) or []
                    b = new_entry.get(key) or []
                    try:
                        cur[key] = sorted({str(x) for x in a} | {str(x) for x in b})
                    except Exception:
                        cur[key] = a or b
                # instances: append unique dicts by (worksheet,row_index,title)
                inst_a = cur.get("instances") or []
                inst_b = new_entry.get("instances") or []
                seen = {(i.get("worksheet"), i.get("row_index"), i.get("title")) for i in inst_a}
                for i in inst_b:
                    sig = (i.get("worksheet"), i.get("row_index"), i.get("title"))
                    if sig not in seen:
                        inst_a.append(i)
                        seen.add(sig)
                cur["instances"] = inst_a
        unique_entries = list(dedup.values())
        semaphore = asyncio.Semaphore(self.config.concurrency)
        per_domain: Dict[str, asyncio.Semaphore] = {}
        results: List[FetchResult] = []

        connector = aiohttp.TCPConnector(limit=self.config.concurrency)
        headers = {
            "User-Agent": self.config.user_agent,
            "Accept-Language": self.config.accept_language,
            "Accept-Encoding": "gzip, deflate",
        }

        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            # Phase 1: fetch provided entries
            tasks: List[asyncio.Task] = []
            for entry in unique_entries:
                task = asyncio.create_task(self._fetch_entry(entry, session, semaphore, per_domain))
                setattr(task, "_sparta_entry", entry)
                tasks.append(task)
            total = len(tasks)
            completed = 0
            for task in asyncio.as_completed(tasks):
                entry = getattr(task, "_sparta_entry", {})
                result = await task
                results.append(result)
                completed += 1
                if progress_hook is not None:
                    try:
                        progress_hook(completed, total, entry, result)
                    except Exception:
                        pass

            # Phase 2 (optional): hub fan-out – fetch child links discovered on link-hub pages
            try:
                if os.getenv("SPARTA_TOC_FANOUT", "1") == "1":
                    # Build a set of already-requested normalized URLs
                    existing: set[str] = set()
                    for e in unique_entries:
                        existing.add(_normalize_url(e.get("url") or ""))
                    for r in results:
                        existing.add(_normalize_url(r.url))

                    fanout_entries: List[Dict[str, Any]] = []
                    # Fan-out telemetry (kept/dropped + reasons)
                    toc_drop_examples: List[Dict[str, str]] = []
                    toc_drop_reasons: Dict[str, int] = {}
                    toc_children_kept = 0
                    toc_children_dropped = 0
                    for r in results:
                        meta = r.metadata or {}
                        if not meta or not meta.get("link_hub"):
                            continue
                        links = meta.get("link_hub_links") or []
                        if not isinstance(links, list) or not links:
                            continue
                        # Inherit key metadata (controls/titles/frameworks/worksheets/instances/contexts) from the hub entry
                        inherit_keys = {"controls", "titles", "frameworks", "worksheets", "instances", "contexts"}
                        inherited: Dict[str, Any] = {k: meta.get(k) for k in inherit_keys if meta.get(k) is not None}
                        inherited["parent_url"] = r.url
                        inherited["source"] = "hub_fanout"
                        # Optional path allowlist (domain-specific)
                        allowlist_map = _load_path_allowlist()
                        base_host = urlparse(r.url).hostname or ""
                        allowed_prefixes = allowlist_map.get(base_host.lower()) or []
                        for link in links:
                            norm = _normalize_url(link)
                            if not norm or norm in existing:
                                continue
                            # Apply allowlist when configured for this domain
                            if allowed_prefixes:
                                p = urlparse(norm).path or "/"
                                if not any(p.startswith(pref) for pref in allowed_prefixes):
                                    toc_children_dropped += 1
                                    reason = "path_not_allowed"
                                    toc_drop_reasons[reason] = toc_drop_reasons.get(reason, 0) + 1
                                    if len(toc_drop_examples) < 8:
                                        toc_drop_examples.append({"url": norm, "reason": reason})
                                    continue
                            entry = {"url": norm, **inherited}
                            fanout_entries.append(entry)
                            existing.add(norm)
                    # Fetch fan-out children (bounded by link extractor itself)
                    if fanout_entries:
                        tasks2 = [asyncio.create_task(self._fetch_entry(e, session, semaphore, per_domain)) for e in fanout_entries]
                        children_results: List[FetchResult] = []
                        for future in asyncio.as_completed(tasks2):
                            children_results.append(await future)
                        # Centralized prefilters (skip weak children before Step 07)
                        kept_children: List[FetchResult] = []
                        for cr in children_results:
                            # Only filter fan-out children
                            if (cr.metadata or {}).get("source") != "hub_fanout":
                                kept_children.append(cr)
                                continue
                            # Apply centralized prefilters with heading density + per-domain overrides
                            decision = evaluate_body_prefilter(
                                text=cr.text or "",
                                content_type=cr.content_type or "text/html",
                                url=cr.url,
                            )
                            if not decision.keep:
                                toc_children_dropped += 1
                                reason = decision.reason or "prefilter_reject"
                                toc_drop_reasons[reason] = toc_drop_reasons.get(reason, 0) + 1
                                if len(toc_drop_examples) < 8:
                                    toc_drop_examples.append({"url": cr.url, "reason": reason})
                                # Annotate telemetry on dropped rows too
                                md = {**(cr.metadata or {})}
                                md["prefilter_checked"] = True
                                md["prefilter_reason"] = reason
                                if decision.section_pref_fallback:
                                    md["section_pref_fallback"] = decision.section_pref_fallback
                                cr.metadata = md
                                continue
                            toc_children_kept += 1
                            md = {**(cr.metadata or {})}
                            md["prefilter_checked"] = True
                            md["prefilter_reason"] = None
                            if decision.section_pref_fallback:
                                md["section_pref_fallback"] = decision.section_pref_fallback
                            cr.metadata = md
                            kept_children.append(cr)
                        results.extend(kept_children)
                    # Attach fan-out telemetry to an internal attribute for audit merging
                    self._last_fanout_stats = {
                        "toc_children_kept": toc_children_kept,
                        "toc_children_dropped": toc_children_dropped,
                        "toc_drop_reasons": toc_drop_reasons,
                        "toc_drop_examples": toc_drop_examples,
                    }
            except Exception:
                # Hub fan-out is best-effort; never fail the batch
                pass

        if self._cache_dirty and self.cache_path:
            await self._flush_cache_to_disk()

        extra = getattr(self, "_last_fanout_stats", None)
        audit = self._build_audit(results, len(unique_entries), extra=extra)
        return results, audit

    async def _flush_cache_to_disk(self) -> None:
        """Prune the in-memory HTTP cache and persist to disk (bounded)."""
        async with self._cache_lock:
            try:
                cap = int(os.getenv("SPARTA_HTTP_CACHE_MAX_ENTRIES", "5000"))
            except Exception:
                cap = 5000
            if cap > 0 and len(self._cache) > cap:
                def _ts(v: Dict[str, Any]) -> str:
                    return v.get("fetched_at", "") or ""
                items = sorted(self._cache.items(), key=lambda kv: _ts(kv[1]))
                drop = len(items) - cap
                for k, _ in items[:max(0, drop)]:
                    self._cache.pop(k, None)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")

    async def _fetch_entry(
        self,
        entry: Dict[str, Any],
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        per_domain: Dict[str, asyncio.Semaphore],
    ) -> FetchResult:
        original_url = entry["url"].strip()
        redirect_reason: Optional[str] = None
        redirected = self._apply_static_redirects(original_url)
        url = redirected or original_url
        if redirected:
            entry["url"] = url
        url_norm = _normalize_url(url)
        parsed = urlparse(url_norm)
        domain = parsed.netloc.lower()
        if redirected:
            redirect_reason = "static_redirect"
        base_metadata = {
            key: value
            for key, value in entry.items()
            if key not in {"url", "domain", "text", "status"}
        }
        if redirected:
            base_metadata["original_url"] = original_url
        if url_norm != url:
            base_metadata["url_normalized"] = url_norm
        def _ensure_redirect(meta: Dict[str, Any]) -> Dict[str, Any]:
            if redirect_reason and "redirect_reason" not in meta:
                meta["redirect_reason"] = redirect_reason
            return meta

        domain_sem = per_domain.setdefault(domain, asyncio.Semaphore(self.config.per_domain))

        async with semaphore, domain_sem:
            # Use normalized key for cache; fall back to raw for legacy entries
            cached = self._cache.get(url_norm) or self._cache.get(url)
            if cached:
                cache_meta = cached.get("metadata") or {}
                # Retro-fit link hub analysis for cached HTML if missing
                try:
                    force_recompute = os.getenv("SPARTA_TOC_RECOMPUTE", "0") == "1"
                    if (force_recompute or ("link_hub" not in cache_meta)) and (cached.get("content_type", "").startswith("text/html") or (cached.get("content_type") in {None, ""})):
                        hub_meta = self._analyze_link_hub(cached.get("text", "") or "", url)
                        if hub_meta:
                            cache_meta.update(hub_meta)
                            # Write back updated metadata to cache atomically
                            cached["metadata"] = cache_meta
                            async with self._cache_lock:
                                self._cache[url_norm] = cached
                                self._cache_dirty = True
                except Exception:
                    pass
                metadata = _ensure_redirect({**base_metadata, **cache_meta})
                return FetchResult(
                    url=url,
                    domain=domain,
                    status=cached["status"],
                    content_type=cached.get("content_type", "text/html"),
                    text=cached.get("text", ""),
                    fetched_at=cached.get("fetched_at", ""),
                    method=cached.get("method", "cache"),
                    metadata=metadata,
                    from_cache=True,
                    raw_bytes=None,
                )

            if domain == "d3fend.mitre.org":
                local_result = self._fetch_d3fend_local(parsed, base_metadata)
                if local_result is not None:
                    status, content_type, text, method, extra_metadata, local_bytes = local_result
                    fetched_at = datetime.now(timezone.utc).isoformat()
                    metadata = _ensure_redirect({**base_metadata, **(extra_metadata or {})})
                    if status == 200 and text:
                        async with self._cache_lock:
                            self._cache[url_norm] = {
                                "status": status,
                                "content_type": content_type,
                                "text": text,
                                "fetched_at": fetched_at,
                                "method": method,
                                "metadata": extra_metadata,
                            }
                            self._cache_dirty = True
                    return FetchResult(
                        url=url,
                        domain=domain,
                        status=status,
                        content_type=content_type,
                        text=text,
                        fetched_at=fetched_at,
                        method=method,
                        metadata=metadata,
                        from_cache=False,
                        raw_bytes=local_bytes,
                    )

            try:
                if parsed.scheme == "file":
                    status, content_type, text, method, extra_metadata, local_bytes = self._fetch_from_file(parsed)
                    fetched_at = datetime.now(timezone.utc).isoformat()
                    metadata = _ensure_redirect({**base_metadata, **(extra_metadata or {})})
                    return FetchResult(
                        url=url,
                        domain=domain,
                        status=status,
                        content_type=content_type,
                        text=text,
                        fetched_at=fetched_at,
                        method=method,
                        metadata=metadata,
                        from_cache=False,
                        raw_bytes=local_bytes,
                    )
                github_request = github_utils.prepare_github_request(url, url_norm)
                if github_request is not None:
                    github_result = await self._fetch_github_entry(
                        github_request,
                        base_metadata,
                        session,
                    )
                    if github_result is not None:
                        return github_result
                status, content_type, text, method, extra_metadata, raw_bytes = await self._fetch_with_retries(
                    session, url_norm, domain
                )
                fetched_at = datetime.now(timezone.utc).isoformat()
                # If trailing-slash variant might help (common 404 pitfall), try once
                tried_variant = False
                if (status in {404, 400, 405} or (status == 200 and not text)) and (parsed.path or ""):
                    alt = _toggle_trailing_slash(url_norm)
                    if alt and alt != url_norm:
                        try:
                            status2, content_type2, text2, method2, extra_meta2, raw_bytes2 = await self._fetch_with_retries(
                                session, alt, domain
                            )
                            tried_variant = True
                            if status2 == 200 and text2:
                                status, content_type, text, method = status2, content_type2, text2, method2
                                raw_bytes = raw_bytes2
                                extra_metadata = {**extra_metadata, **extra_meta2, "trailing_slash_fallback": True, "url_variant_used": alt}
                        except Exception:
                            pass
                if status == 200 and text:
                    self._cache[url_norm] = {
                        "status": status,
                        "content_type": content_type,
                        "text": text,
                        "fetched_at": fetched_at,
                        "method": method,
                        "metadata": extra_metadata,
                    }
                    self._cache_dirty = True
                error = None
                metadata = _ensure_redirect({**base_metadata, **(extra_metadata or {})})
            except Exception as exc:  # pragma: no cover - runtime failure path
                status = -1
                content_type = "error"
                text = ""
                fetched_at = datetime.now(timezone.utc).isoformat()
                method = "error"
                error = str(exc)
                metadata = _ensure_redirect(dict(base_metadata))
                if status in {404, 410, -1}:
                    hint = self._brave_lookup(url)
                    if hint:
                        metadata["brave_suggestion"] = hint
                raw_bytes = None

            return FetchResult(
                url=url,
                domain=domain,
                status=status,
                content_type=content_type,
                text=text,
                fetched_at=fetched_at,
                method=method,
                metadata=metadata,
                from_cache=False,
                error=error,
                raw_bytes=raw_bytes,
            )

    async def _fetch_github_entry(
        self,
        request: github_utils.GithubRequest,
        base_metadata: Dict[str, Any],
        session: aiohttp.ClientSession,
    ) -> Optional[FetchResult]:
        try:
            status, content_type, text, method, extra_metadata, raw_bytes = await self._fetch_with_retries(
                session,
                request.fetch_url,
                request.domain,
                extra_headers=request.headers,
            )
        except Exception:
            status = -1
            content_type = "error"
            text = ""
            method = "github_raw"
            extra_metadata = {}
            raw_bytes = None

        if status != 200:
            cli_result = await github_utils.fetch_with_cli(request)
            if cli_result is None:
                return None
            status, content_type, text, method, extra_metadata, raw_bytes = cli_result

        fetched_at = datetime.now(timezone.utc).isoformat()
        cache_metadata = {**request.metadata, **(extra_metadata or {})}
        metadata = {**base_metadata, **cache_metadata}
        metadata.setdefault("github_fetch_url", request.fetch_url)
        result = FetchResult(
            url=request.original_url,
            domain=urlparse(request.original_url).netloc.lower(),
            status=status,
            content_type=content_type,
            text=text,
            fetched_at=fetched_at,
            method=method,
            metadata=metadata,
            from_cache=False,
            raw_bytes=raw_bytes,
        )

        if status == 200 and text:
            async with self._cache_lock:
                self._cache[request.normalized_url] = {
                    "status": status,
                    "content_type": content_type,
                    "text": text,
                    "fetched_at": fetched_at,
                    "method": method,
                    "metadata": cache_metadata,
                }
                self._cache_dirty = True
        return result

    async def _fetch_with_retries(
        self,
        session: aiohttp.ClientSession,
        url: str,
        domain: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]:
        delay = self.config.backoff_initial
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await self._fetch_once(session, url, domain, extra_headers=extra_headers)
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt == self.config.max_attempts:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.backoff_max)
        if last_exc:
            raise last_exc
        raise RuntimeError("unexpected retry state")

    async def _fetch_once(
        self,
        session: aiohttp.ClientSession,
        url: str,
        domain: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]:
        try:
            ssl_context = None
            if domain in self._insecure_ssl_domains:
                ssl_context = False  # type: ignore[assignment]
            request_headers = extra_headers or None
            async with session.get(
                url,
                timeout=self.config.timeout,
                ssl=ssl_context,
                headers=request_headers,
            ) as resp:
                status = resp.status
                content_type = resp.headers.get("Content-Type", "text/html").split(";")[0]
                encoding = (resp.headers.get("Content-Encoding") or "").lower()
                raw_bytes = await resp.read()
        except asyncio.TimeoutError:
            raise

        method = "aiohttp"
        extra_metadata: Dict[str, Any] = {}
        if self._insecure_ssl_domains and domain in self._insecure_ssl_domains:
            extra_metadata["ssl_verification"] = False
        text = ""
        body_bytes: Optional[bytes] = raw_bytes

        is_pdf = "pdf" in content_type.lower() or url.lower().endswith(".pdf")

        if is_pdf and raw_bytes:
            method = "pdf"
            text, pdf_meta = self._extract_pdf_text(raw_bytes, url)
            extra_metadata.update(pdf_meta)
            return 200, "application/pdf", text, method, extra_metadata, raw_bytes

        if "zstd" in encoding:
            import zstandard as zstd  # type: ignore

            try:
                raw_bytes = zstd.ZstdDecompressor().decompress(raw_bytes)
            except zstd.ZstdError:  # pragma: no cover - corrupted payload
                raw_bytes = b""

        if raw_bytes:
            text = raw_bytes.decode("utf-8", "ignore")
            body_bytes = raw_bytes

        fallback_statuses = {401, 403, 404, 429}
        fallback_allowed = async_playwright is not None
        domain_requires_spa = domain in SPA_FALLBACK_DOMAINS
        needs_playwright = False

        if status == 200 and not is_pdf:
            needs_playwright = domain_requires_spa or self._needs_playwright(url, text, content_type, domain)
            if needs_playwright and fallback_allowed:
                status, content_type, text, pw_meta, pw_bytes = await self._fetch_with_playwright(url, status)
                method = "playwright"
                extra_metadata.update(pw_meta)
                body_bytes = pw_bytes
            # After we have final HTML text, analyze for link-hub/ToC pages
            try:
                hub_meta = self._analyze_link_hub(text or "", url)
                if hub_meta:
                    extra_metadata.update(hub_meta)
            except Exception:
                pass
        elif fallback_allowed and (domain_requires_spa or status in fallback_statuses):
            status_playwright, content_type_playwright, text_playwright, pw_meta, pw_bytes = await self._fetch_with_playwright(url, status)
            if text_playwright:
                status = status_playwright or status
                content_type = content_type_playwright
                text = text_playwright
                method = "playwright"
                extra_metadata.update(pw_meta)
                body_bytes = pw_bytes
                # Analyze link-hub characteristics for the Playwright-rendered HTML
                try:
                    hub_meta = self._analyze_link_hub(text or "", url)
                    if hub_meta:
                        extra_metadata.update(hub_meta)
                except Exception:
                    pass

        # Certain providers (e.g., Cloudflare) return 200 with a block page.
        block_tokens = ("cf-error-details", "you are unable to access", "you have been blocked")
        lower_text = (text or "").lower()
        need_wayback = False
        if status == 200 and text and any(token in lower_text for token in block_tokens):
            need_wayback = True
        elif status in {401, 403, 404}:
            if (not text) or any(token in lower_text for token in block_tokens):
                need_wayback = True

        if need_wayback:
            wayback = await self._fetch_from_wayback(session, url)
            if wayback is not None:
                status, content_type, text, method, wb_meta, wb_bytes = wayback
                extra_metadata.update(wb_meta)
                body_bytes = wb_bytes
            else:
                jina = await self._fetch_from_jina(session, url)
                if jina is not None:
                    status, content_type, text, method, wb_meta, wb_bytes = jina
                    extra_metadata.update(wb_meta)
                    body_bytes = wb_bytes

        if status in {404, 410, -1}:
            brave_hint = self._brave_lookup(url)
            if brave_hint:
                extra_metadata.setdefault("brave_suggestion", brave_hint)
                alt_status = await self._attempt_alternate(brave_hint, session)
                if alt_status is not None:
                    alt_status, alt_content_type, alt_text, alt_method, alt_meta, alt_bytes = alt_status
                    status = alt_status
                    content_type = alt_content_type
                    text = alt_text
                    method = alt_method
                    extra_metadata.update(alt_meta)
                    extra_metadata.setdefault("alternate_note", "brave_redirect")
                    body_bytes = alt_bytes

        return status, content_type, text, method, extra_metadata, body_bytes

    def _brave_lookup(self, url: str) -> Optional[str]:
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            return None
        try:
            query = self._build_brave_query(url)
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": 5},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": api_key,
                },
                timeout=5,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None
        results = data.get("web", {}).get("results", []) if isinstance(data, dict) else []
        for item in results:
            candidate = item.get("url")
            if not candidate or candidate == url:
                continue
            parsed = urlparse(candidate)
            if parsed.scheme in {"http", "https"} and parsed.netloc:
                return candidate
        return None

    def _build_brave_query(self, url: str) -> str:
        parsed = urlparse(url)
        parts: List[str] = []
        if parsed.path:
            tail = parsed.path.rstrip("/").split("/")[-1]
            tail = tail.replace(".html", "").replace(".pdf", "")
            tokens = re.findall(r"[A-Za-z]+", tail)
            if tokens:
                parts.append(" ".join(tokens))
        if parsed.netloc:
            parts.append(parsed.netloc.replace("www.", ""))
        query = " ".join(parts).strip()
        return query or url

    async def _attempt_alternate(
        self,
        alt_url: str,
        session: aiohttp.ClientSession,
    ) -> Optional[Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]]:
        try:
            return await self._fetch_with_retries(session, alt_url, urlparse(alt_url).netloc.lower())
        except Exception:
            return None

    def _apply_static_redirects(self, url: str) -> Optional[str]:
        if "d3f:CredentialRevoking" in url:
            return url.replace("CredentialRevoking", "CredentialRevocation")
        return None

    async def _fetch_from_wayback(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> Optional[Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]]:
        """Attempt to retrieve the resource from the Internet Archive.

        Returns ``None`` when no archival snapshot is available.
        """

        cdx_url = (
            "https://web.archive.org/cdx/search/cdx?url="
            f"{quote(url, safe='')}&output=json&limit=1&filter=statuscode:200&collapse=digest"
        )
        try:
            async with session.get(cdx_url, timeout=10) as resp:
                if resp.status != 200:
                    return None
                try:
                    candidates = await resp.json()
                except Exception:
                    raw = await resp.text()
                    try:
                        candidates = json.loads(raw)
                    except Exception:
                        return None
        except Exception:
            return None

        timestamp = _select_wayback_timestamp(candidates)
        if not timestamp:
            return None
        archive_url = f"https://web.archive.org/web/{timestamp}/{url}"
        try:
            async with session.get(archive_url, timeout=self.config.timeout) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.headers.get("Content-Type", "text/html").split(";")[0]
                body = await resp.read()
        except Exception:
            return None

        text = body.decode("utf-8", "ignore") if body else ""
        metadata = {
            "via": "wayback",
            "wayback_timestamp": timestamp,
        }
        return 200, content_type, text, "wayback", metadata, body

    async def _fetch_from_jina(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> Optional[Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]]:
        """Fallback using r.jina.ai proxy when Cloudflare blocks direct access."""

        if not url.startswith("http"):
            return None
        jina_url = "https://r.jina.ai/" + url
        try:
            async with session.get(jina_url, timeout=self.config.timeout) as resp:
                if resp.status != 200:
                    return None
                body = await resp.read()
        except Exception:
            return None

        text = body.decode("utf-8", "ignore") if body else ""
        if not text.strip():
            return None
        metadata = {
            "via": "jina",
            "jina_url": jina_url,
        }
        return 200, "text/html", text, "jina", metadata, body

    def _analyze_link_hub(self, html: str, base_url: str) -> Dict[str, Any]:
        """Detect link-heavy ToC/portal pages and harvest outlinks (bounded).

        Emits deterministic metadata only; downstream steps decide whether to enqueue.
        """
        if not html or not base_url:
            return {}
        # Heuristics (env-tunable)
        links_min = int(os.getenv("SPARTA_TOC_LINKS_MIN", "20"))
        body_chars_max = int(os.getenv("SPARTA_TOC_BODY_CHARS_MAX", "1200"))
        link_density_min = float(os.getenv("SPARTA_TOC_LINK_DENSITY_MIN", "0.4"))
        links_cap = int(os.getenv("SPARTA_TOC_LINKS_MAX", "100"))
        same_domain_only = os.getenv("SPARTA_TOC_SAME_DOMAIN_ONLY", "1") == "1"
        allow_domains_env = os.getenv("SPARTA_TOC_ALLOWED_DOMAINS", "")
        allow_domains = {d.strip().lower() for d in allow_domains_env.split(",") if d.strip()}

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        # Remove common nav/footer containers if present
        for css in ("nav", "footer", "header", "aside"):
            for t in soup.find_all(css):
                t.decompose()
        body_text = soup.get_text(separator="\n") if soup else ""
        body_text_chars = len(body_text.strip())

        anchors = soup.find_all("a") if soup else []
        links: List[str] = []
        link_text_chars = 0
        base_host = urlparse(base_url).hostname or ""
        # Optional path allowlist mapping (domain -> [prefixes])
        allowlist_map = _load_path_allowlist()
        allowed_prefixes = allowlist_map.get(base_host.lower()) or []
        dropped_examples: List[Dict[str, str]] = []
        dropped_count = 0
        for a in anchors:
            href = (a.get("href") or "").strip()
            if not href:
                continue
            # skip in-page anchors and javascript:void(0)
            if href.startswith("#") or href.startswith("javascript:"):
                continue
            absu = urljoin(base_url, href)
            parsed = urlparse(absu)
            if parsed.scheme not in {"http", "https"}:
                continue
            host = parsed.hostname or ""
            # Filter obvious non-content links
            path = (parsed.path or "").strip()
            if not path or path == "/":
                dropped_count += 1
                if len(dropped_examples) < 6:
                    dropped_examples.append({"url": absu, "reason": "root_path"})
                continue
            if "cdn-cgi" in path or "email-protection" in path:
                dropped_count += 1
                if len(dropped_examples) < 6:
                    dropped_examples.append({"url": absu, "reason": "cdn_cgi"})
                continue
            if same_domain_only and host.lower() != base_host.lower():
                # allow-list domains override same_domain_only
                if host.lower() not in allow_domains:
                    continue
            # Apply path allowlist if configured for this domain
            if allowed_prefixes:
                if not any(path.startswith(pref) for pref in allowed_prefixes):
                    dropped_count += 1
                    if len(dropped_examples) < 6:
                        dropped_examples.append({"url": absu, "reason": "path_not_allowed"})
                    continue
            links.append(_normalize_url(absu))
            link_text_chars += len((a.get_text(" ") or "").strip())

        # Dedup and cap links
        seen = set()
        uniq_links: List[str] = []
        for u in links:
            if u not in seen:
                uniq_links.append(u)
                seen.add(u)
            if len(uniq_links) >= links_cap:
                break

        link_count = len(uniq_links)
        link_density = (link_text_chars / max(1, body_text_chars)) if body_text_chars else 0.0
        reasons: List[str] = []
        if link_count >= links_min:
            reasons.append("links_min")
        if body_text_chars <= body_chars_max:
            reasons.append("low_body_text")
        if link_density >= link_density_min:
            reasons.append("high_link_density")

        is_link_hub = len([r for r in reasons if r in {"links_min", "high_link_density"}]) >= 1 and (link_count >= links_min)
        meta: Dict[str, Any] = {
            "link_hub": bool(is_link_hub),
            "link_hub_reasons": reasons,
            "link_hub_links": uniq_links,
            "link_hub_count": link_count,
            "link_hub_body_chars": body_text_chars,
            "link_hub_link_text_chars": link_text_chars,
            "hub_title": (soup.title.string.strip() if soup and soup.title and soup.title.string else None),
            "link_hub_dropped_count": dropped_count,
            "link_hub_drop_examples": dropped_examples,
        }
        return meta

    def _fetch_d3fend_local(
        self,
        parsed_url: Any,
        base_metadata: Dict[str, Any],
    ) -> Optional[Tuple[int, str, str, str, Dict[str, Any], bytes]]:
        if self._local_data_root is None:
            return None

        path = parsed_url.path.strip("/")
        if not path:
            return None

        segments = [segment for segment in path.split("/") if segment]
        if not segments:
            return None

        dataset = None
        identifier = None

        if len(segments) >= 3 and segments[0] == "dao" and segments[1] == "artifact":
            dataset = "d3fend_artifacts.csv"
            identifier = segments[2]
        elif len(segments) >= 2 and segments[0] == "technique":
            dataset = "d3fend_techniques.csv"
            identifier = segments[1]
        elif len(segments) >= 2 and segments[0] == "tactic":
            dataset = "d3fend_tactics.csv"
            identifier = segments[1]

        if dataset is None or identifier is None:
            return None

        identifier = unquote(identifier.rstrip("/"))
        if not identifier:
            return None

        record = self._lookup_d3fend_record(dataset, identifier)
        if record is None:
            return None

        lines: List[str] = []
        for key, value in record.items():
            if value is None or value == "":
                continue
            if isinstance(value, str) and value.strip() == "nan":
                continue
            lines.append(f"{key}: {value}")
        if not lines:
            return None

        text = "\n".join(lines) + "\n"
        metadata = {
            "d3fend_dataset": dataset,
            "d3fend_id": identifier,
            "source": "d3fend_local",
        }
        return 200, "text/plain", text, "d3fend_local", metadata, text.encode("utf-8")

    def _lookup_d3fend_record(self, dataset: str, identifier: str) -> Optional[Dict[str, str]]:
        if dataset not in self._d3fend_cache:
            catalog = {}
            data_path = self._local_data_root / "data" / "raw" / dataset
            if not data_path.exists():
                self._d3fend_cache[dataset] = catalog
            else:
                with data_path.open("r", encoding="utf-8") as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        normalized_row = {
                            column: (value or "").strip()
                            for column, value in row.items()
                        }
                        key_candidates = set()
                        for candidate_key in ("ID", "id", "@id"):
                            value = normalized_row.get(candidate_key)
                            if value:
                                key_candidates.add(value)
                        url_value = normalized_row.get("URL") or normalized_row.get("Url")
                        if url_value:
                            key_candidates.add(url_value)
                            url_tail = unquote(url_value.rstrip("/").split("/")[-1])
                            if url_tail:
                                key_candidates.add(url_tail)
                        for key in key_candidates:
                            catalog[key] = normalized_row
                self._d3fend_cache[dataset] = catalog
        catalog = self._d3fend_cache.get(dataset) or {}
        return catalog.get(identifier)

    def _needs_playwright(self, url: str, text: str, content_type: str, domain: str) -> bool:
        if not self.config.verify_html:
            return False
        if "html" not in content_type.lower():
            return False
        if domain in SIGNIN_MODAL_DOMAINS:
            return True
        if len(text) < self.config.min_text_length:
            return True
        soup = BeautifulSoup(text, "lxml")
        if not soup.find("body"):
            return True
        body = soup.find("body")
        if body:
            for tag in body.find_all(["script", "style", "noscript"]):
                tag.extract()
            stripped = body.get_text(" ", strip=True)
            if len(stripped) < self.config.js_text_threshold:
                return True
        assessment = extract_content_features(text, content_type, url)
        if assessment.text_len < self.config.js_text_threshold:
            return True
        # Detect common JS placeholders
        js_indicators = [
            "please enable javascript",
            "javascript is required",
            "</noscript>",
            "__NEXT_DATA__",
            "id=\"root\"",
        ]
        lowered = text.lower()
        return any(indicator in lowered for indicator in js_indicators)

    async def _fetch_with_playwright(self, url: str, origin_status: Optional[int] = None) -> Tuple[int, str, str, Dict[str, Any], bytes]:
        assert async_playwright is not None  # For type checker
        async with async_playwright() as p:  # type: ignore
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(user_agent=self.config.user_agent)
            page = await context.new_page()
            dismissed_modal = False
            screenshot_path: Optional[Path] = None
            try:
                response = await page.goto(
                    url,
                    timeout=int(self.config.timeout * 1000),
                    wait_until="domcontentloaded",
                )
                await page.wait_for_timeout(2000)
                domain = urlparse(url).netloc.lower()
                dismissed_modal = await self._maybe_dismiss_signin_modal(page, domain)
                if dismissed_modal:
                    await page.wait_for_timeout(500)
                if self.config.screenshots_dir:
                    try:
                        self.config.screenshots_dir.mkdir(parents=True, exist_ok=True)
                        sha = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
                        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                        screenshot_path = self.config.screenshots_dir / f"{sha}-{ts}.png"
                        await page.screenshot(path=str(screenshot_path), full_page=True)
                    except Exception:
                        screenshot_path = None
                content = await page.content()
                status = response.status if response else 0
                content_type = response.headers.get("content-type", "text/html") if response else "text/html"
            finally:
                await context.close()
                await browser.close()
        metadata: Dict[str, Any] = {"playwright": True}
        if origin_status is not None:
            metadata["fallback_from_status"] = origin_status
        if dismissed_modal:
            metadata["signin_modal_dismissed"] = True
        if screenshot_path is not None:
            metadata["playwright_screenshot"] = str(screenshot_path)
        content_type_final = content_type.split(";")[0]
        return status, content_type_final, content, metadata, content.encode("utf-8", "ignore")

    async def _maybe_dismiss_signin_modal(self, page, domain: str) -> bool:
        """Attempt to close intrusive sign-in overlays for known domains."""

        if domain not in SIGNIN_MODAL_DOMAINS:
            return False

        selectors = [
            "button[aria-label='Dismiss']",
            "button.artdeco-modal__dismiss",
            "button[data-test-modal-close-btn]",
            "button[aria-label='Dismiss alert']",
        ]
        for selector in selectors:
            try:
                await page.locator(selector).first.click(timeout=1500)
                return True
            except Exception:
                continue

        try:
            await page.evaluate(
                "() => {"
                "document.querySelectorAll('.artdeco-modal, .artdeco-modal__overlay, .modal').forEach(el => el.remove());"
                "document.querySelectorAll('.modal__backdrop, .artdeco-toast-item').forEach(el => el.remove());"
                "}"
            )
            return True
        except Exception:
            return False

    def _fetch_from_file(self, parsed_url: Any) -> Tuple[int, str, str, str, Dict[str, Any], bytes]:
        def _is_within(child: Path, root: Path) -> bool:
            try:
                child.resolve().relative_to(root.resolve())
                return True
            except Exception:
                return False
        local_path = Path(parsed_url.path)
        if not local_path.is_absolute():
            local_path = Path(parsed_url.netloc + parsed_url.path).resolve()
        base = self._local_data_root or Path.cwd()
        if not _is_within(local_path, base):
            raise PermissionError(f"Refusing file access outside allowed base: {local_path}")
        if not local_path.exists():
            raise FileNotFoundError(f"Local artifact not found: {local_path}")

        raw_bytes = local_path.read_bytes()
        suffix = local_path.suffix.lower()

        if suffix == ".pdf":
            text, pdf_meta = self._extract_pdf_text(raw_bytes, str(local_path))
            metadata = {"local_path": str(local_path), **pdf_meta, "playwright": False}
            return 200, "application/pdf", text, "file", metadata, raw_bytes
        if suffix in {".html", ".htm"}:
            text = raw_bytes.decode("utf-8", "ignore")
            return 200, "text/html", text, "file", {"local_path": str(local_path)}, raw_bytes
        # default to plain text
        text = raw_bytes.decode("utf-8", "ignore")
        return 200, "text/plain", text, "file", {"local_path": str(local_path)}, raw_bytes

    def _extract_pdf_text(self, raw_bytes: bytes, url: str) -> Tuple[str, Dict[str, Any]]:
        try:
            doc = fitz.open(stream=raw_bytes, filetype="pdf")
        except Exception as exc:
            raise RuntimeError(f"PDF open failed for {url}: {exc}") from exc
        try:
            text_parts: List[str] = []
            for page in doc:
                page_text = page.get_text("text") or ""
                if page_text:
                    text_parts.append(page_text.strip())
            full_text = "\n".join(part for part in text_parts if part)

            used_ocr = False
            if not full_text.strip():
                # Try to recover text via words extraction first
                word_text = []
                for page in doc:
                    words = page.get_text("words") or []
                    if words:
                        ordered = sorted(words, key=lambda w: (w[3], w[0]))
                        builder: List[str] = []
                        last_y = None
                        for x0, y0, x1, y1, word, *_ in ordered:
                            if last_y is not None and abs(y0 - last_y) > 2.5:
                                builder.append("\n")
                            builder.append(word)
                            builder.append(" ")
                            last_y = y0
                        text = "".join(builder).strip()
                        if text:
                            word_text.append(text)
                if word_text:
                    full_text = "\n".join(word_text)

            if not full_text.strip() and pytesseract is not None and Image is not None:
                ocr_text: List[str] = []
                matrix = fitz.Matrix(2, 2)
                for page in doc:
                    pix = page.get_pixmap(matrix=matrix)
                    mode = "RGB"
                    if pix.n >= 4:
                        mode = "RGBA"
                    if Image is None:
                        continue
                    image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                    ocr_result = pytesseract.image_to_string(image)
                    if ocr_result:
                        ocr_text.append(ocr_result.strip())
                if ocr_text:
                    full_text = "\n".join(ocr_text)
                    used_ocr = True

            if not full_text.strip():
                raise RuntimeError(f"PDF extraction empty for {url}")

            metadata = {
                "pdf_text_extracted": True,
                "pdf_pages": doc.page_count,
                "pdf_characters": len(full_text),
            }
            if used_ocr:
                metadata["pdf_ocr_fallback"] = True
            return full_text, metadata
        finally:
            doc.close()

    def _build_audit(self, results: List[FetchResult], target_total: int, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        successes = sum(1 for r in results if r.status == 200 and has_text_payload(r))
        failures = sum(1 for r in results if r.status != 200)
        audit = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "requested": target_total,
            "fetched": successes,
            "failed": failures,
            "cache_hits": sum(1 for r in results if r.from_cache),
        }
        if extra:
            audit.update({
                "toc_children_kept": int(extra.get("toc_children_kept", 0)),
                "toc_children_dropped": int(extra.get("toc_children_dropped", 0)),
                "toc_drop_reasons": extra.get("toc_drop_reasons", {}),
                "toc_drop_examples": extra.get("toc_drop_examples", []),
            })
        return audit

# ---------- Helpers for path allowlists (domain-specific) ----------
def _load_path_allowlist() -> Dict[str, List[str]]:
    """Load domain->path-prefix allowlists from env, with safe defaults.

    Env:
      - SPARTA_TOC_PATH_ALLOWLIST_JSON: JSON mapping {"host":["/prefix/", ...]}
      - SPARTA_TOC_ALLOWLIST_DISABLE_DEFAULTS=1 to disable built-in defaults
    """
    mapping: Dict[str, List[str]] = {}
    raw = os.getenv("SPARTA_TOC_PATH_ALLOWLIST_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                for host, prefixes in data.items():
                    if not isinstance(host, str) or not isinstance(prefixes, (list, tuple)):
                        continue
                    cleaned = [str(p).strip() for p in prefixes if str(p).strip()]
                    if cleaned:
                        mapping[host.lower()] = cleaned
        except Exception:
            pass
    if os.getenv("SPARTA_TOC_ALLOWLIST_DISABLE_DEFAULTS", "0") != "1":
        # Safe defaults for CSRC NIST hubs
        mapping.setdefault("csrc.nist.gov", [
            "/projects/",
            "/presentations/",
            "/publications/",
            "/news",
        ])
    return mapping


def write_results(results: List[FetchResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(result.to_json() + "\n")


def _select_wayback_timestamp(candidates: Any) -> Optional[str]:
    """Extract the first valid timestamp from a CDX response payload."""

    if not isinstance(candidates, list):
        return None
    # First row is a header; iterate over remaining rows until we find a valid stamp
    for row in candidates[1:]:
        if not isinstance(row, list) or not row:
            continue
        stamp = str(row[0]).strip()
        if stamp:
            return stamp
    return None
