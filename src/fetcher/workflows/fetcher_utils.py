"""Shared helper functions used by the fetcher workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set, TYPE_CHECKING

from ..core.keys import K_TEXT_PATH

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .web_fetch import FetchResult


def idna_normalize(host: str) -> str:
    """Return a lowercase, IDNA-normalized host name."""

    h = (host or "").strip().rstrip(".").lower()
    if not h:
        return ""
    try:
        h = h.encode("idna").decode("ascii")
    except Exception:
        pass
    return h


def normalize_domain(domain: str, strip_subdomains: Set[str]) -> str:
    """Normalize a domain, optionally stripping well-known subdomains."""

    d = idna_normalize(domain)
    if not d:
        return d
    labels = d.split(".")
    if len(labels) > 2 and labels[0] in strip_subdomains:
        return ".".join(labels[1:])
    return d


def normalize_set(domains: Iterable[str], strip: Set[str]) -> Set[str]:
    """Normalize every domain in an iterable using :func:`normalize_domain`."""

    return {normalize_domain(d, strip) for d in domains}


def is_safe_domain(
    domain: str,
    safe_domains: Set[str],
    safe_suffixes: Iterable[str],
    strip_subdomains: Set[str] | None = None,
) -> bool:
    """Return True when the domain matches an explicit list or suffix."""

    strip = strip_subdomains or set()
    normalized = normalize_domain(domain, strip)
    if not normalized:
        return False
    if normalized in safe_domains:
        return True
    for suffix in safe_suffixes:
        token = (suffix or "").strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        if normalized.endswith(token):
            return True
    return False


def resolve_repo_root(inventory_path: Path) -> Path:
    """Climb ancestors until a ``data/processed`` dir is found."""

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


def has_text_payload(result: "FetchResult | None") -> bool:
    """Return True if a fetch result still has inline text or an external blob."""

    if result is None:
        return False
    if getattr(result, "text", None):
        return True
    metadata = getattr(result, "metadata", None) or {}
    return bool(metadata.get(K_TEXT_PATH))


def sanity_check() -> None:
    assert idna_normalize("ExAmple.COM") == "example.com"
    assert normalize_domain("www.example.com", {"www"}) == "example.com"
    assert "example.com" in normalize_set(["www.EXAMPLE.com"], {"www"})
    assert is_safe_domain("test.example.gov", {"example.com"}, (".gov",), {"www"})
    assert not is_safe_domain("medium.com", set(), (".gov",), {"www"})


sanity_check()

__all__ = [
    "idna_normalize",
    "normalize_domain",
    "normalize_set",
    "is_safe_domain",
    "resolve_repo_root",
    "has_text_payload",
    "sanity_check",
]
