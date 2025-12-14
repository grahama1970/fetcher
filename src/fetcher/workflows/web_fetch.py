from __future__ import annotations

import asyncio
from asyncio.subprocess import PIPE as SUBPROCESS_PIPE
import csv
import hashlib
import itertools
import re
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import multiprocessing
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Set
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
from .fetcher_config import OVERRIDES_PATH as DEFAULT_OVERRIDES_PATH
from .prefilters import evaluate_body_prefilter
from . import github_utils
from .extract_utils import extract_content_features
from .fetcher_utils import has_text_payload

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

logger = logging.getLogger(__name__)

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


def _pdf_crack_settings() -> Dict[str, Any]:
    def _bool(name: str, default: str = "0") -> bool:
        raw = os.getenv(name, default)
        return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}

    def _int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, str(default)))
        except Exception:
            return default

    charset = os.getenv("FETCHER_PDF_CRACK_CHARSET", "0123456789")
    processes_default = max(1, (os.cpu_count() or 2) // 2)

    return {
        "enable": _bool("FETCHER_PDF_CRACK_ENABLE", "0"),
        "charset": charset,
        "minlen": _int("FETCHER_PDF_CRACK_MINLEN", 4),
        "maxlen": _int("FETCHER_PDF_CRACK_MAXLEN", 6),
        "processes": max(1, _int("FETCHER_PDF_CRACK_PROCESSES", processes_default)),
        "timeout": max(1, _int("FETCHER_PDF_CRACK_TIMEOUT", 15)),
        "verbose": _bool("FETCHER_PDF_CRACK_VERBOSE", "0"),
        "bruteforce_limit": max(0, _int("FETCHER_PDF_BRUTE_LIMIT", 50000)),
    }


def _brute_force_pdf_password(
    raw_bytes: bytes,
    charset: str,
    minlen: int,
    maxlen: int,
    limit: int,
) -> Tuple[Optional[str], Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "pdf_password_bruteforce_attempted": True,
        "pdf_password_bruteforce_limit": limit,
    }
    if limit <= 0:
        meta["pdf_password_bruteforce_skipped"] = "disabled"
        return None, meta
    if fitz is None:
        meta["pdf_password_bruteforce_error"] = "pymupdf_missing"
        return None, meta
    charset = "".join(dict.fromkeys(charset))
    if not charset:
        meta["pdf_password_bruteforce_error"] = "empty_charset"
        return None, meta

    total_space = 0
    for length in range(max(1, minlen), max(minlen, maxlen) + 1):
        try:
            total_space += len(charset) ** length
        except OverflowError:
            total_space = limit + 1
            break
    meta["pdf_password_bruteforce_space"] = total_space
    if limit > 0 and total_space > limit:
        meta["pdf_password_bruteforce_skipped"] = "limit_exceeded"
        return None, meta

    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        meta["pdf_password_bruteforce_error"] = f"open_failed:{exc}"
        return None, meta

    attempts = 0
    try:
        for length in range(max(1, minlen), max(minlen, maxlen) + 1):
            for combo in itertools.product(charset, repeat=length):
                attempts += 1
                candidate = "".join(combo)
                try:
                    if doc.authenticate(candidate):
                        meta["pdf_password_bruteforce_attempts"] = attempts
                        return candidate, meta
                except Exception:
                    continue
    finally:
        doc.close()
    meta["pdf_password_bruteforce_attempts"] = attempts
    meta["pdf_password_bruteforce_failed"] = True
    return None, meta


def _crack_pdf_password(raw_bytes: bytes, url: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Attempt to crack a password-protected PDF using pdferli.

    Returns (password | None, metadata). Metadata may include timeout/error flags.
    This is best-effort and bounded by FETCHER_PDF_CRACK_TIMEOUT seconds.
    """

    settings = _pdf_crack_settings()
    meta: Dict[str, Any] = {
        "pdf_password_crack_enabled": bool(settings.get("enable")),
        "pdf_password_charset": settings.get("charset"),
        "pdf_password_minlen": settings.get("minlen"),
        "pdf_password_maxlen": settings.get("maxlen"),
        "pdf_password_processes": settings.get("processes"),
    }

    if not settings.get("enable"):
        meta["pdf_password_crack_skipped"] = "disabled"
        return None, meta

    try:
        from pdferli import crack_password  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        meta["pdf_password_crack_error"] = f"pdferli_unavailable: {exc}"
        return None, meta

    charset = str(settings.get("charset") or "")
    if not charset:
        meta["pdf_password_crack_error"] = "empty_charset"
        return None, meta

    timeout = int(settings.get("timeout", 15))
    minlen = int(settings.get("minlen", 1))
    maxlen = int(settings.get("maxlen", minlen))
    processes = int(settings.get("processes", 1))
    verbose = bool(settings.get("verbose"))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    def _worker(path: str, queue: multiprocessing.Queue) -> None:
        try:
            pwd = crack_password(
                file=path,
                chars=charset,
                minlen=minlen,
                maxlen=maxlen,
                processes=processes,
                verbose=verbose,
            )
            queue.put({"password": pwd})
        except Exception as exc:  # pragma: no cover - runtime failure
            queue.put({"error": str(exc)})

    queue: multiprocessing.Queue = multiprocessing.Queue()
    start = time.perf_counter()
    proc = multiprocessing.Process(target=_worker, args=(tmp_path, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        meta["pdf_password_crack_timeout"] = True
        meta["pdf_password_crack_ms"] = int((time.perf_counter() - start) * 1000)
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return None, meta

    result = queue.get() if not queue.empty() else {}
    meta["pdf_password_crack_ms"] = int((time.perf_counter() - start) * 1000)

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if "error" in result:
        meta["pdf_password_crack_error"] = result.get("error")
        return None, meta

    password = result.get("password")
    if not password:
        limit = int(settings.get("bruteforce_limit", 0))
        fallback_password: Optional[str] = None
        fallback_meta: Dict[str, Any] = {}
        if limit != 0:
            fallback_password, fallback_meta = _brute_force_pdf_password(
                raw_bytes,
                charset,
                minlen,
                maxlen,
                limit,
            )
            meta.update(fallback_meta)
        if fallback_password:
            password = fallback_password
        else:
            meta["pdf_password_crack_failed"] = True
            return None, meta

    meta["pdf_password_recovered"] = True
    meta["pdf_password_length"] = len(str(password))
    meta["pdf_password_charset_used"] = settings.get("charset")
    return str(password), meta

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

    - Avoid rewriting path characters (path-sensitive sites can 404)
    - Lower-case host
    - Remove default ports
    """
    try:
        raw = (u or "").strip()
        p = urlparse(raw)
        host = p.hostname.lower() if p.hostname else None
        path = p.path or ""
        if path and any(ch.isspace() for ch in path):
            path = "".join(path.split())
        netloc = host if host else p.netloc
        # Drop default ports
        if p.port and ((p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443)):
            netloc = host or netloc
        p = p._replace(netloc=netloc, path=path)
        return urlunparse(p)
    except Exception:
        return u or ""


_TARGET_LOCAL_OVERRIDE_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _load_target_local_overrides(path: Path) -> List[Dict[str, Any]]:
    key = str(path.resolve())
    if key in _TARGET_LOCAL_OVERRIDE_CACHE:
        return _TARGET_LOCAL_OVERRIDE_CACHE[key]
    rules: List[Dict[str, Any]] = []
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                rules = data
    except Exception:
        rules = []
    _TARGET_LOCAL_OVERRIDE_CACHE[key] = rules
    return rules


def _has_target_local_override(url: str, overrides_path: Optional[Path]) -> bool:
    """Return True if overrides.json declares a target_local_file for this URL."""
    env_path = os.getenv("FETCHER_OVERRIDES_PATH")
    path = Path(env_path) if env_path else (overrides_path or Path(DEFAULT_OVERRIDES_PATH))
    overrides_path = path
    rules = _load_target_local_overrides(overrides_path)
    if not rules:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    for rule in rules:
        local_file = rule.get("target_local_file")
        if not local_file:
            continue
        domain = str(rule.get("domain") or "").strip().lower()
        if domain and domain != (parsed.netloc or "").lower():
            continue
        path_prefix = str(rule.get("path_prefix") or "").strip()
        if path_prefix and not (parsed.path or "").startswith(path_prefix):
            continue
        substr = str(rule.get("substring") or "").strip()
        if substr and substr not in url:
            continue
        return True
    return False

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

def _looks_like_github_pages_404(text: str) -> bool:
    """Detect the GitHub Pages soft-404 template that (wrongly) returns 200."""
    if not text:
        return False
    lowered = text.lower()
    return (
        "page not found \u00b7 github pages" in lowered
        or "the site configured at this address" in lowered
        or "does not contain the requested file" in lowered
    )


_SOFT_404_TEMPLATES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "github_pages_404",
        (
            "page not found \u00b7 github pages",
            "the site configured at this address",
            "does not contain the requested file",
        ),
    ),
    (
        "cloudflare_access_denied",
        (
            "cloudflare",
            "error 1020",
            "access denied",
            "you are unable to access",
        ),
    ),
    (
        "s3_nosuchkey",
        (
            "<code>nosuchkey</code>",
            "<code>nosuchkey",
            "code>nosuchkey<",
        ),
    ),
    (
        "generic_not_found",
        (
            "page not found",
            "requested url was not found on this server",
            "the page you are looking for doesn't exist",
        ),
    ),
)


def _detect_soft_404(text: str) -> Optional[str]:
    """Return a template id when the body matches a known soft-404/blocked page."""
    if not text:
        return None
    lowered = text.lower()
    for template_id, tokens in _SOFT_404_TEMPLATES:
        if all(token in lowered for token in tokens):
            return template_id
    if _looks_like_github_pages_404(text):
        return "github_pages_404"
    # Fallback: single-token generic match for short bodies
    if "page not found" in lowered and "requested url" in lowered:
        return "generic_not_found"
    return None


def _split_env_list(value: str, *, lower: bool = False) -> Tuple[str, ...]:
    tokens: List[str] = []
    for token in value.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        tokens.append(cleaned.lower() if lower else cleaned)
    # Preserve order but drop duplicates
    seen: Set[str] = set()
    ordered: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return tuple(ordered)


def _env_int_list(raw: str) -> Tuple[int, ...]:
    values: List[int] = []
    for token in raw.split(","):
        cleaned = token.strip()
        if not cleaned:
            continue
        try:
            values.append(int(cleaned))
        except ValueError:
            continue
    # Preserve order but dedupe
    deduped: List[int] = []
    seen: Set[int] = set()
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        deduped.append(val)
    return tuple(deduped)


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if not normalized:
        return default
    return normalized not in {"0", "false", "off", "no"}


def _safe_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        cleaned = value.strip()
        if not cleaned:
            return default
        return int(cleaned)
    except Exception:
        return default


def _detect_proxy_credit_exhaustion(status: int, text: Optional[str], error: Optional[str]) -> Optional[str]:
    haystacks: List[str] = []
    if text:
        haystacks.append(text)
    if error:
        haystacks.append(error)
    for blob in haystacks:
        lower = blob.lower()
        for keyword in PROXY_CREDIT_KEYWORDS:
            if keyword in lower:
                return keyword
    if status in PROXY_CREDIT_STATUS_HINTS:
        return f"status_{status}"
    return None


DEFAULT_PROXY_DOMAINS = ("d3fend.mitre.org",)
DEFAULT_PROXY_STATUSES = (429,)
DEFAULT_PROXY_HINTS = (
    "rate limit",
    "too many requests",
    "temporarily blocked",
    "retry later",
    "exceeded your request quota",
    "too many hits",
)

PROXY_CREDIT_STATUS_HINTS = {402, 407, 509, 511}
PROXY_CREDIT_KEYWORDS = (
    "out of credit",
    "out-of-credit",
    "out of credits",
    "no credit",
    "insufficient credit",
    "insufficient credits",
    "insufficient balance",
    "insufficient traffic",
    "traffic exhausted",
    "bandwidth exhausted",
    "bandwidth limit exceeded",
    "not enough credit",
    "exhausted your traffic",
)


@dataclass(frozen=True)
class ProxyRotationSettings:
    scheme: str
    host: str
    port: int
    username: Optional[str]
    password: Optional[str]
    provider: str
    allowed_domains: Tuple[str, ...] = ()
    trigger_statuses: Tuple[int, ...] = DEFAULT_PROXY_STATUSES
    trigger_hints: Tuple[str, ...] = DEFAULT_PROXY_HINTS

    @property
    def proxy_url(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    @property
    def display_endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    def allows_domain(self, domain: str) -> bool:
        if not self.allowed_domains:
            return True
        domain = domain.lower()
        for token in self.allowed_domains:
            token = token.strip().lstrip(".")
            if not token:
                continue
            if token == "*":
                return True
            if domain == token or domain.endswith(f".{token}"):
                return True
        return False

    def should_proxy(self, domain: str, status: int, text: Optional[str]) -> Optional[str]:
        if not self.allows_domain(domain):
            return None
        if self.trigger_statuses and status in self.trigger_statuses:
            return f"status_{status}"
        if text:
            lower_text = text.lower()
            for hint in self.trigger_hints:
                token = hint.strip().lower()
                if not token:
                    continue
                if token in lower_text:
                    return f"hint_{token}"
        return None


def _load_proxy_rotation_from_env() -> Optional[ProxyRotationSettings]:
    if _as_bool(os.getenv("SPARTA_STEP06_PROXY_DISABLE"), False):
        return None

    raw_url = os.getenv("SPARTA_STEP06_PROXY_URL", "").strip()
    scheme = "http"
    host = ""
    port: Optional[int] = None
    username: Optional[str] = None
    password: Optional[str] = None

    if raw_url:
        parsed = urlparse(raw_url if "://" in raw_url else f"http://{raw_url}")
        scheme = parsed.scheme or scheme
        host = parsed.hostname or host
        port = parsed.port or port
        if parsed.username:
            username = parsed.username
        if parsed.password:
            password = parsed.password

    host = host or os.getenv("SPARTA_STEP06_PROXY_HOST", os.getenv("IPROYAL_HOST", "")).strip()
    port = port or _safe_int(os.getenv("SPARTA_STEP06_PROXY_PORT", os.getenv("IPROYAL_PORT", "")))

    creds = os.getenv("SPARTA_STEP06_PROXY_CREDENTIALS", "").strip()
    if creds and (":" in creds) and (not username or not password):
        user_part, _, pwd_part = creds.partition(":")
        username = username or user_part
        password = password or pwd_part

    username = username or os.getenv(
        "SPARTA_STEP06_PROXY_USER",
        os.getenv("SPARTA_STEP06_PROXY_USERNAME", os.getenv("IPROYAL_USER", "")),
    ).strip()
    password = password or os.getenv(
        "SPARTA_STEP06_PROXY_PASSWORD",
        os.getenv("SPARTA_STEP06_PROXY_PASS", os.getenv("IPROYAL_PASSWORD", os.getenv("IPROYAL_PASS", ""))),
    ).strip()

    provider = os.getenv("SPARTA_STEP06_PROXY_PROVIDER", "iproyal").strip() or "iproyal"
    domain_tokens = _split_env_list(os.getenv("SPARTA_STEP06_PROXY_DOMAINS", ""), lower=True)
    if not domain_tokens and not _as_bool(os.getenv("SPARTA_STEP06_PROXY_DOMAINS_DISABLE_DEFAULTS")):
        domain_tokens = DEFAULT_PROXY_DOMAINS
    status_tokens = _env_int_list(os.getenv("SPARTA_STEP06_PROXY_STATUSES", ""))
    if not status_tokens:
        status_tokens = DEFAULT_PROXY_STATUSES
    hint_tokens = _split_env_list(os.getenv("SPARTA_STEP06_PROXY_HINTS", ""), lower=True)
    if not hint_tokens:
        hint_tokens = DEFAULT_PROXY_HINTS

    if not host or port is None:
        return None

    username = username or None
    password = password or None

    return ProxyRotationSettings(
        scheme=scheme or "http",
        host=host,
        port=int(port),
        username=username,
        password=password,
        provider=provider,
        allowed_domains=domain_tokens,
        trigger_statuses=status_tokens,
        trigger_hints=hint_tokens,
    )


def _is_iproyal_provider(settings: Optional[ProxyRotationSettings]) -> bool:
    if settings is None:
        return False
    return settings.provider.lower().startswith("iproyal")


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
    proxy_rotation: Optional[ProxyRotationSettings] = None
    disable_http_cache: bool = False
    overrides_path: Optional[Path] = None
    # Controls hub fan-out (Phase 2). When None, fall back to env SPARTA_TOC_FANOUT (default on).
    enable_toc_fanout: Optional[bool] = None


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
        text_length = len(self.text)
        metadata = self.metadata or {}
        if text_length == 0:
            try:
                text_length = int(metadata.get("text_length_chars", 0))
            except Exception:
                text_length = 0
        payload = {
            "url": self.url,
            "domain": self.domain,
            "status": self.status,
            "content_type": self.content_type,
            "text_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest()
            if self.text
            else "",
            "text_length": text_length,
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
        env_disable_cache = os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1"
        self._cache_enabled = not (config.disable_http_cache or env_disable_cache)
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self._cache_enabled and cache_path and cache_path.exists():
            try:
                self._cache = json.loads(cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._cache = {}
        self._cache_dirty = False
        # Guard cache mutations across async tasks sharing the same event loop
        self._cache_lock: asyncio.Lock = asyncio.Lock()
        # Resolve overrides path: env takes precedence, then config, then default policy path.
        env_overrides = os.getenv("FETCHER_OVERRIDES_PATH")
        if env_overrides:
            self._overrides_path = Path(env_overrides)
        elif config.overrides_path is not None:
            self._overrides_path = config.overrides_path
        else:
            self._overrides_path = Path(DEFAULT_OVERRIDES_PATH)
        self._local_data_root = (
            Path(config.local_data_root).resolve()
            if config.local_data_root is not None
            else None
        )
        self._d3fend_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._insecure_ssl_domains = {domain.lower() for domain in config.insecure_ssl_domains}
        self._proxy_rotation: Optional[ProxyRotationSettings] = config.proxy_rotation or _load_proxy_rotation_from_env()
        self._proxy_audit: Optional[Dict[str, Any]] = None

    async def fetch_many(
        self,
        entries: Iterable[Dict[str, Any]],
        progress_hook: Optional[Callable[[int, int, Dict[str, Any], FetchResult], None]] = None,
    ) -> Tuple[List[FetchResult], Dict[str, Any]]:
        self._proxy_audit = self._init_proxy_audit()
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
        start_time = time.perf_counter()
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
                enable_fanout = self.config.enable_toc_fanout
                if enable_fanout is None:
                    enable_fanout = os.getenv("SPARTA_TOC_FANOUT", "1") == "1"
                if enable_fanout:
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
                    domain_allowances = _load_domain_allowlist()
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
                        base_host_lower = base_host.lower()
                        allowed_prefixes = allowlist_map.get(base_host_lower) or []
                        domain_allow = _domain_matches_allowlist(base_host_lower, domain_allowances)
                        for link in links:
                            norm = _normalize_url(link)
                            if not norm or norm in existing:
                                continue
                            parsed_link = urlparse(norm)
                            link_host = (parsed_link.hostname or "").lower()
                            if link_host and link_host not in {base_host_lower} and domain_allow:
                                if not _domain_matches_allowlist(link_host, {domain_allow}):
                                    toc_children_dropped += 1
                                    reason = "domain_not_allowed"
                                    toc_drop_reasons[reason] = toc_drop_reasons.get(reason, 0) + 1
                                    if len(toc_drop_examples) < 8:
                                        toc_drop_examples.append({"url": norm, "reason": reason})
                                    continue
                            # Apply allowlist when configured for this domain
                            if allowed_prefixes:
                                p = parsed_link.path or "/"
                                if not any(p.startswith(pref) for pref in allowed_prefixes):
                                    toc_children_dropped += 1
                                    reason = "path_not_allowed"
                                    toc_drop_reasons[reason] = toc_drop_reasons.get(reason, 0) + 1
                                    if len(toc_drop_examples) < 8:
                                        toc_drop_examples.append({"url": norm, "reason": reason})
                                    continue
                            elif domain_allow:
                                if not link_host or not _domain_matches_allowlist(link_host, {domain_allow}):
                                    toc_children_dropped += 1
                                    reason = "domain_not_allowed"
                                    toc_drop_reasons[reason] = toc_drop_reasons.get(reason, 0) + 1
                                    if len(toc_drop_examples) < 8:
                                        toc_drop_examples.append({"url": norm, "reason": reason})
                                    continue
                            else:
                                # No allowlist and domain not approved; skip
                                toc_children_dropped += 1
                                reason = "no_allowlist"
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

        if self._cache_enabled and self._cache_dirty and self.cache_path:
            await self._flush_cache_to_disk()

        runtime = max(0.001, time.perf_counter() - start_time)
        rate_metrics = _summarize_rate_limits(results, runtime)
        extra = getattr(self, "_last_fanout_stats", {}) or {}
        extra["rate_limit_metrics"] = rate_metrics
        if self._proxy_audit is not None:
            extra["proxy_rotation"] = self._proxy_audit
        audit = _build_audit(results, len(unique_entries), extra=extra)
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
            cached = None
            if self._cache_enabled:
                cached = self._cache.get(url_norm) or self._cache.get(url)
                # Skip cached entries when a target_local_file override exists so mirrors win.
                try:
                    if cached and _has_target_local_override(url, self._overrides_path):
                        cached = None
                except Exception:
                    cached = cached
                # Skip cached d3fend_local entries when local data root is disabled
                local_root_disabled = (self._local_data_root is None) or (os.getenv("SPARTA_STEP06_DISABLE_LOCAL_DATA_ROOT", "0") == "1") or (os.getenv("FETCHER_DISABLE_LOCAL_DATA_ROOT", "0") == "1")
                if cached and cached.get("method") == "d3fend_local" and local_root_disabled:
                    cached = None
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
                    if self._cache_enabled and status == 200 and text:
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
                proxy_reason = self._should_use_proxy(domain, status, text)
                if proxy_reason:
                    proxy_fetch, proxy_meta = await self._fetch_with_proxy(
                        session,
                        url_norm,
                        domain,
                        proxy_reason,
                    )
                    extra_metadata = {**extra_metadata, **proxy_meta}
                    if proxy_fetch is not None:
                        status, content_type, text, method, proxy_extra_metadata, proxy_bytes = proxy_fetch
                        raw_bytes = proxy_bytes
                        extra_metadata = {**extra_metadata, **proxy_extra_metadata}
                    else:
                        extra_metadata.setdefault("proxy_rotation_failed", True)
                if self._cache_enabled and status == 200 and text:
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
        proxy_settings: Optional[ProxyRotationSettings] = None,
        proxy_reason: Optional[str] = None,
    ) -> Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]:
        delay = self.config.backoff_initial
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await self._fetch_once(
                    session,
                    url,
                    domain,
                    extra_headers=extra_headers,
                    proxy_settings=proxy_settings,
                    proxy_reason=proxy_reason,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt == self.config.max_attempts:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.backoff_max)
        if last_exc:
            raise last_exc
        raise RuntimeError("unexpected retry state")

    def _init_proxy_audit(self) -> Optional[Dict[str, Any]]:
        if not self._proxy_rotation:
            return None
        return {
            "enabled": True,
            "provider": self._proxy_rotation.provider,
            "endpoint": self._proxy_rotation.display_endpoint,
            "attempted": 0,
            "success": 0,
            "failed": 0,
            "domains": {},
            "reasons": {},
            "status_counts": {},
        }

    def _proxy_audit_attempt(self, domain: str, reason: str) -> None:
        if self._proxy_audit is None:
            return
        self._proxy_audit["attempted"] = int(self._proxy_audit.get("attempted", 0)) + 1
        domains = self._proxy_audit.setdefault("domains", {})
        domains[domain] = int(domains.get(domain, 0)) + 1
        reasons = self._proxy_audit.setdefault("reasons", {})
        reasons[reason] = int(reasons.get(reason, 0)) + 1

    def _proxy_audit_result(self, status: int, *, success: bool) -> None:
        if self._proxy_audit is None:
            return
        key = "success" if success else "failed"
        self._proxy_audit[key] = int(self._proxy_audit.get(key, 0)) + 1
        counts = self._proxy_audit.setdefault("status_counts", {})
        counts[str(status)] = int(counts.get(str(status), 0)) + 1

    def _proxy_audit_error(self, domain: str, reason: str, url: str, error: str) -> None:
        if self._proxy_audit is None:
            return
        self._proxy_audit["failed"] = int(self._proxy_audit.get("failed", 0)) + 1
        errors = self._proxy_audit.setdefault("errors", [])
        errors.append({
            "domain": domain,
            "reason": reason,
            "url": url,
            "error": error,
        })

    def _proxy_audit_credit(self, domain: str, url: str, trigger: str, detail: str, status: Optional[int] = None) -> None:
        if self._proxy_audit is None:
            return
        self._proxy_audit["credit_exhausted"] = int(self._proxy_audit.get("credit_exhausted", 0)) + 1
        events = self._proxy_audit.setdefault("credit_events", [])
        if len(events) < 20:
            events.append({
                "domain": domain,
                "url": url,
                "trigger": trigger,
                "detail": detail,
                "status": status,
            })

    def _flag_proxy_credit_exhaustion(
        self,
        metadata: Dict[str, Any],
        domain: str,
        url: str,
        trigger_reason: str,
        detail: str,
        status: Optional[int],
    ) -> None:
        metadata["proxy_rotation_credit_exhausted"] = True
        metadata["proxy_rotation_credit_reason"] = detail
        metadata.setdefault("proxy_rotation_error", "proxy_credit_exhausted")
        if status is not None:
            metadata["proxy_rotation_credit_status"] = status
        self._proxy_audit_credit(domain, url, trigger_reason, detail, status)
        logger.warning(
            "Proxy rotation credits exhausted via %s for %s (%s): %s",
            (self._proxy_rotation.provider if self._proxy_rotation else "proxy"),
            domain,
            url,
            detail,
        )

    def _should_use_proxy(self, domain: str, status: int, text: Optional[str]) -> Optional[str]:
        if not self._proxy_rotation:
            return None
        return self._proxy_rotation.should_proxy(domain, status, text)

    async def _fetch_with_proxy(
        self,
        session: aiohttp.ClientSession,
        url: str,
        domain: str,
        reason: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]], Dict[str, Any]]:
        settings = self._proxy_rotation
        metadata: Dict[str, Any] = {
            "proxy_rotation_attempted": True,
            "proxy_rotation_reason": reason,
            "proxy_rotation_provider": settings.provider if settings else None,
            "proxy_rotation_endpoint": settings.display_endpoint if settings else None,
        }
        if not settings:
            metadata["proxy_rotation_error"] = "proxy_disabled"
            return None, metadata
        self._proxy_audit_attempt(domain, reason)
        try:
            result = await self._fetch_with_retries(
                session,
                url,
                domain,
                extra_headers=extra_headers,
                proxy_settings=settings,
                proxy_reason=reason,
            )
        except Exception as exc:  # pragma: no cover - runtime-specific failures
            error_message = str(exc)
            metadata["proxy_rotation_error"] = error_message
            if _is_iproyal_provider(settings):
                credit_detail = _detect_proxy_credit_exhaustion(-1, None, error_message)
                if credit_detail:
                    self._flag_proxy_credit_exhaustion(metadata, domain, url, reason, credit_detail, None)
            self._proxy_audit_error(domain, reason, url, error_message)
            return None, metadata
        status = result[0]
        success = status == 200
        self._proxy_audit_result(status, success=success)
        if _is_iproyal_provider(settings):
            credit_detail = _detect_proxy_credit_exhaustion(status, result[2], None)
            if credit_detail:
                self._flag_proxy_credit_exhaustion(metadata, domain, url, reason, credit_detail, status)
        return result, metadata

    async def _fetch_once(
        self,
        session: aiohttp.ClientSession,
        url: str,
        domain: str,
        extra_headers: Optional[Dict[str, str]] = None,
        proxy_settings: Optional[ProxyRotationSettings] = None,
        proxy_reason: Optional[str] = None,
    ) -> Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]:
        try:
            ssl_context = None
            if domain in self._insecure_ssl_domains:
                ssl_context = False  # type: ignore[assignment]
            request_headers = extra_headers or None
            request_kwargs: Dict[str, Any] = {}
            if proxy_settings is not None:
                request_kwargs["proxy"] = proxy_settings.proxy_url
                if proxy_settings.username or proxy_settings.password:
                    request_kwargs["proxy_auth"] = aiohttp.BasicAuth(
                        proxy_settings.username or "",
                        proxy_settings.password or "",
                    )
            async with session.get(
                url,
                timeout=self.config.timeout,
                ssl=ssl_context,
                headers=request_headers,
                **request_kwargs,
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
        if proxy_settings is not None:
            extra_metadata["proxy_rotation_used"] = True
            extra_metadata["proxy_rotation_reason"] = proxy_reason
            extra_metadata["proxy_rotation_provider"] = proxy_settings.provider
            extra_metadata["proxy_rotation_endpoint"] = proxy_settings.display_endpoint
        text = ""
        body_bytes: Optional[bytes] = raw_bytes

        is_pdf = "pdf" in content_type.lower() or url.lower().endswith(".pdf")
        pdf_parse_failed = False
        if is_pdf:
            pdf_bytes: Optional[bytes] = None
            if (
                status == 200
                and raw_bytes
                and ("pdf" in content_type.lower() or raw_bytes.startswith(b"%PDF"))
            ):
                pdf_bytes = raw_bytes
            else:
                fallback = await self._fetch_pdf_via_wget(url)
                if fallback is not None:
                    pdf_bytes, wget_meta = fallback
                    extra_metadata.update(wget_meta)
                    status = 200
                    method = "wget"
            if pdf_bytes:
                method = method if method == "wget" else "pdf"
                try:
                    text, pdf_meta = self._extract_pdf_text(pdf_bytes, url)
                    extra_metadata.update(pdf_meta)
                    return 200, "application/pdf", text, method, extra_metadata, pdf_bytes
                except RuntimeError as exc:
                    extra_metadata["pdf_parse_error"] = str(exc)
                    pdf_parse_failed = True
                    status = 404
                    text = ""
                    body_bytes = None
                    method = "error"
            else:
                pdf_parse_failed = True
                status = 404
                text = ""
                body_bytes = None
                method = "error"

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
                original_text = text
                original_bytes = body_bytes
                try:
                    status, content_type, text, pw_meta, pw_bytes = await self._fetch_with_playwright(url, status)
                    method = "playwright"
                    extra_metadata.update(pw_meta)
                    if text:
                        body_bytes = pw_bytes
                    else:
                        text = original_text
                        body_bytes = original_bytes
                        extra_metadata.setdefault("playwright_failed", True)
                except Exception as exc:  # pragma: no cover - runtime fallback
                    text = original_text
                    body_bytes = original_bytes
                    extra_metadata.setdefault("playwright_failed", True)
                    extra_metadata.setdefault("playwright_error", str(exc))
            elif needs_playwright and not fallback_allowed:
                # When a page is JS-heavy but Playwright is unavailable, fall back to
                # r.jina.ai which performs remote rendering/extraction.
                extra_metadata.setdefault("playwright_unavailable", True)
                jina = await self._fetch_from_jina(session, url)
                if jina is not None:
                    status, content_type, text, method, jina_meta, jina_bytes = jina
                    extra_metadata.update(jina_meta)
                    body_bytes = jina_bytes
            # After we have final HTML text, analyze for link-hub/ToC pages
            try:
                hub_meta = self._analyze_link_hub(text or "", url)
                if hub_meta:
                    extra_metadata.update(hub_meta)
            except Exception:
                pass
        elif fallback_allowed and not is_pdf and (domain_requires_spa or status in fallback_statuses):
            original_text = text
            original_bytes = body_bytes
            try:
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
                else:
                    text = original_text
                    body_bytes = original_bytes
                    extra_metadata.setdefault("playwright_failed", True)
            except Exception as exc:  # pragma: no cover - runtime fallback
                text = original_text
                body_bytes = original_bytes
                extra_metadata.setdefault("playwright_failed", True)
                extra_metadata.setdefault("playwright_error", str(exc))

        if status == 200 and "html" in (content_type or "").lower():
            soft_404_id = _detect_soft_404(text or "")
            if soft_404_id:
                extra_metadata["soft_404_detected"] = True
                extra_metadata["soft_404_template"] = soft_404_id
                status = 404
                text = ""
                body_bytes = None

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

        # Optional Google Translate fallback for stubborn bot/interstitial 403/503 pages.
        if status in {403, 503} and os.getenv("FETCHER_ENABLE_GOOGLE_TRANSLATE", "0") == "1":
            gt = await self._fetch_from_google_translate(session, url)
            if gt is not None:
                status, content_type, text, method, gt_meta, gt_bytes = gt
                extra_metadata.update(gt_meta)
                body_bytes = gt_bytes

        # Final guard: never return 200 when a soft-404 template was detected.
        if extra_metadata.get("soft_404_detected") and status == 200:
            status = 404
            text = ""
            body_bytes = None

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

    async def _fetch_pdf_via_wget(self, url: str) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        cmd = ["wget", "-q", "-O", "-", url]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=SUBPROCESS_PIPE,
                stderr=SUBPROCESS_PIPE,
            )
        except FileNotFoundError:
            return None
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0 or not stdout:
            return None
        metadata = {
            "wget_fallback": True,
            "wget_command": " ".join(cmd),
        }
        if stderr:
            metadata["wget_stderr"] = stderr.decode("utf-8", "ignore")[:500]
        return stdout, metadata

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
        # r.jina.ai returns markdown-like text, not raw HTML.
        return 200, "text/plain", text, "jina", metadata, body

    async def _fetch_from_google_translate(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> Optional[Tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]]:
        """Fallback via Google Translate wrapper (optional, flag gated).

        This is best-effort and only used when explicitly enabled, to avoid
        surprising traffic patterns by default.
        """

        wrapper = (
            "https://translate.google.com/translate?"
            f"sl=auto&tl=en&u={quote(url, safe='')}"
        )
        try:
            async with session.get(wrapper, timeout=self.config.timeout) as resp:
                if resp.status != 200:
                    return None
                body = await resp.read()
        except Exception:
            return None

        text = body.decode("utf-8", "ignore") if body else ""
        if not text.strip():
            return None
        metadata = {
            "via": "google_translate",
            "google_translate_url": wrapper,
        }
        return 200, "text/html", text, "google_translate", metadata, body

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

        # A hub should be meaningfully link-heavy relative to body content; don't
        # treat narrative pages with lots of incidental links as hubs.
        is_link_hub = bool(link_count >= links_min and (body_text_chars <= body_chars_max or link_density >= link_density_min))
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
            headed = os.getenv("FETCHER_PLAYWRIGHT_HEADED", "0") == "1"
            headed_domains_env = os.getenv("FETCHER_PLAYWRIGHT_HEADED_DOMAINS", "")
            headed_domains = {
                d.strip().lower()
                for d in headed_domains_env.split(",")
                if d.strip()
            }
            domain = urlparse(url).netloc.lower()
            if domain in headed_domains:
                headed = True
            browser = await p.chromium.launch(headless=not headed)
            # Align context with a typical desktop browser profile. This reduces
            # false-positive bot detection without attempting to evade provider
            # controls.
            context = await browser.new_context(
                user_agent=self.config.user_agent,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                java_script_enabled=True,
            )
            page = await context.new_page()
            dismissed_modal = False
            screenshot_path: Optional[Path] = None
            interstitial_seen = False
            try:
                response = await page.goto(
                    url,
                    timeout=int(self.config.timeout * 1000),
                    wait_until="domcontentloaded",
                )
                # Allow scripts and potential interstitial challenges a brief
                # window to run before we capture the final HTML.
                try:
                    title = (await page.title()) or ""
                except Exception:
                    title = ""
                lower_title = title.lower()
                if "just a moment" in lower_title or "attention required" in lower_title:
                    interstitial_seen = True
                    await page.wait_for_load_state("networkidle", timeout=10000)
                    await page.wait_for_timeout(2000)
                    # Double-hop: retry once in the same context to pick up any
                    # clearance cookies set by the interstitial.
                    second = await page.goto(
                        url,
                        timeout=int(self.config.timeout * 1000),
                        wait_until="networkidle",
                    )
                    if second is not None:
                        response = second
                else:
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
        if interstitial_seen:
            metadata["interstitial_detected"] = True
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
        # Default to the file's own parent directory when no explicit base is
        # provided so absolute file:// URIs outside the current working
        # directory remain readable (use config.local_data_root to constrain).
        allowed_roots = [self._local_data_root] if self._local_data_root else [local_path.parent]
        if not any(_is_within(local_path, root) for root in allowed_roots):
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
        metadata: Dict[str, Any] = {}
        try:
            # Guard early for encrypted/password-protected PDFs. PyMuPDF exposes
            # `needs_pass` when a password is required before text extraction.
            needs_password = bool(getattr(doc, "needs_pass", False))
            if needs_password:
                metadata.update(
                    {
                        "pdf_encrypted": True,
                        "pdf_password_protected": True,
                    }
                )
                password, crack_meta = _crack_pdf_password(raw_bytes, url)
                metadata.update(crack_meta)
                if password:
                    try:
                        doc.close()
                    except Exception:
                        pass
                    try:
                        doc = fitz.open(stream=raw_bytes, filetype="pdf", password=password)
                    except Exception:
                        doc = fitz.open(stream=raw_bytes, filetype="pdf")
                        if not doc.authenticate(password):
                            metadata.setdefault("pdf_password_crack_error", "authentication_failed")
                            metadata["pdf_text_extracted"] = False
                            return "", metadata
                    metadata["pdf_password_recovered"] = True
                else:
                    metadata["pdf_password_recovered"] = False
                    metadata["pdf_text_extracted"] = False
                    return "", metadata

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

            metadata.update(
                {
                    "pdf_text_extracted": True,
                    "pdf_pages": doc.page_count,
                    "pdf_characters": len(full_text),
                }
            )
            if used_ocr:
                metadata["pdf_ocr_fallback"] = True
            return full_text, metadata
        finally:
            doc.close()

def _build_audit(results: List[FetchResult], target_total: int, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            if extra.get("proxy_rotation"):
                audit["proxy_rotation"] = extra.get("proxy_rotation")
            if extra.get("rate_limit_metrics"):
                audit["rate_limit_metrics"] = extra.get("rate_limit_metrics")
        return audit


def _summarize_rate_limits(results: List[FetchResult], runtime_seconds: float) -> Dict[str, Any]:
    runtime = max(runtime_seconds, 1e-6)
    per_domain: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"requests": 0, "rate_limited": 0})
    for result in results:
        bucket = per_domain[result.domain]
        bucket["requests"] += 1
        if result.status == 429:
            bucket["rate_limited"] += 1
    per_domain_plain: Dict[str, Dict[str, Any]] = {}
    for domain, stats in per_domain.items():
        total = max(1, stats["requests"])
        stats["rate_limited_pct"] = round(stats["rate_limited"] / total, 3)
        stats["rps"] = round(stats["requests"] / runtime, 3)
        per_domain_plain[domain] = dict(stats)
    return {
        "runtime_seconds": round(runtime, 3),
        "effective_rps": round((len(results) or 0) / runtime, 3) if results else 0.0,
        "per_domain": per_domain_plain,
    }

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


def _load_domain_allowlist() -> Set[str]:
    raw = os.getenv("SPARTA_TOC_FANOUT_DOMAINS", "").strip()
    domains: Set[str] = set()
    if raw:
        for token in raw.split(","):
            token = token.strip().lower()
            if token:
                domains.add(token.lstrip("."))
    if os.getenv("SPARTA_TOC_FANOUT_DOMAINS_DISABLE_DEFAULTS", "0") != "1":
        domains.update({"attack.mitre.org", "d3fend.mitre.org", "mitre.org", "nist.gov", "csrc.nist.gov", "sparta.aerospace.org", "aerospace.org"})
    return domains


def _domain_matches_allowlist(host: str, allowed: Iterable[str]) -> Optional[str]:
    if not host:
        return None
    normalized = host.strip().lower().rstrip(".")
    for entry in allowed:
        token = entry.strip().lower().lstrip(".")
        if not token:
            continue
        if normalized == token or normalized.endswith(f".{token}"):
            return token
    return None


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
