"""HTML normalization helpers for Readability-friendly extraction.

This module is deterministic and provider-agnostic. It exists to make
extraction (trafilatura/readability) more reliable across brittle pages.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Optional

import ftfy
from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
from lxml.html.clean import Cleaner

try:  # readability-lxml
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

__all__ = [
    "decode_bytes_auto",
    "minimal_text_fix",
    "repair_markup",
    "clean_conservative",
    "readability_text_len_robust",
    "readability_extract_text_robust",
]

_ZERO_WIDTH = {0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF}
_REMOVE = {0x00, 0x0B, 0x0C}
_C1_TO_SPACE = {cp: " " for cp in range(0x80, 0xA0)}
_TRANSLATE = {**{cp: None for cp in _ZERO_WIDTH | _REMOVE}, **_C1_TO_SPACE}

_CONSERVATIVE_CLEANER = Cleaner(
    scripts=True,
    javascript=True,
    style=True,
    links=False,
    forms=False,
    frames=False,
    embedded=False,
    kill_tags={"noscript", "script", "iframe", "style"},
    safe_attrs_only=False,
    remove_unknown_tags=False,
)


def decode_bytes_auto(body: bytes, headers: Optional[Mapping[str, str]] = None) -> str:
    """Decode HTTP bytes using charset header hints with charset-normalizer fallback."""

    enc = None
    if headers:
        ct = headers.get("content-type", "")
        match = re.search(r"charset=([^\s;]+)", ct, re.I)
        if match:
            enc = match.group(1).strip(' "\'').lower()
    if enc:
        try:
            return body.decode(enc, errors="replace")
        except Exception:
            pass
    result = from_bytes(body).best()
    if result is None:
        return body.decode("utf-8", errors="replace")
    return result.output_text


def minimal_text_fix(text: str) -> str:
    """Fix mojibake and strip zero-width/control noise without collapsing structure."""

    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    fixed = ftfy.fix_text(normalized, normalization="NFC")
    return fixed.translate(_TRANSLATE)


def repair_markup(html: str) -> str:
    """Repair malformed markup by parsing/re-serializing with tolerant parsers."""

    for parser in ("html5lib", "lxml", "html.parser"):
        try:
            return str(BeautifulSoup(html, parser))
        except Exception:
            continue
    return html


def clean_conservative(html: str) -> str:
    try:
        return _CONSERVATIVE_CLEANER.clean_html(html)
    except Exception:
        return html


def _readability_extract_text(html: str) -> str:
    if Document is None:
        return ""
    # Strip control chars (0x00-0x1F) that can break readability's XML parser.
    html = html.translate(str.maketrans("", "", "".join(chr(i) for i in range(0x20))))
    if not html:
        return ""
    doc = Document(html)
    snippet = doc.summary(html_partial=True)
    text = BeautifulSoup(snippet, "lxml").get_text(" ", strip=True)
    return text or ""


def readability_extract_text_robust(html: str) -> str:
    """Run Readability with staged fallbacks to avoid crashes on malformed HTML."""

    if not html:
        return ""

    def _attempt(payload: str) -> str:
        if not payload:
            return ""
        try:
            return _readability_extract_text(payload)
        except Exception:
            return ""

    # 1) raw html
    text = _attempt(html)
    if text:
        return text

    fixed = minimal_text_fix(html)
    text = _attempt(fixed)
    if text:
        return text

    repaired = repair_markup(fixed)
    text = _attempt(repaired)
    if text:
        return text

    cleaned = clean_conservative(repaired)
    text = _attempt(cleaned)
    if text:
        return text

    # Last resort: heuristic content region selection.
    try:
        soup = BeautifulSoup(repaired, "lxml")
    except Exception:
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            return ""

    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            continue

    selectors = [
        "article",
        "main",
        "[role='main']",
        "[class*='content']",
        "[id*='content']",
        "[class*='article']",
        "[id*='article']",
        "[class*='post']",
        "[id*='post']",
    ]
    for sel in selectors:
        try:
            nodes = soup.select(sel)
        except Exception:
            continue
        for node in nodes[:8]:
            candidate = node.get_text(" ", strip=True)
            if candidate:
                return candidate

    # Fallback to concatenated paragraph text.
    paras = []
    for node in soup.find_all("p")[:400]:
        txt = node.get_text(" ", strip=True)
        if txt:
            paras.append(txt)
    return " ".join(paras).strip()


def readability_text_len_robust(html: str) -> int:
    """Backwards-compatible helper for paywall heuristics."""

    try:
        return len(readability_extract_text_robust(html) or "")
    except Exception:
        return 0
