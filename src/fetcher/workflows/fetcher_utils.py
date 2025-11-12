"""Shared helper functions used by the fetcher workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set


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


__all__ = [
    "idna_normalize",
    "normalize_domain",
    "normalize_set",
    "resolve_repo_root",
]
