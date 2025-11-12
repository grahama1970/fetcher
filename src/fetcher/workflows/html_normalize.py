"""HTML normalization helpers for paywall detection and preprocessing."""

from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Optional

from bs4 import BeautifulSoup
from charset_normalizer import from_bytes
import ftfy
from lxml.html.clean import Cleaner
from readability import Document  # type: ignore

__all__ = [
    "decode_bytes_auto",
    "minimal_text_fix",
    "repair_markup",
    "clean_conservative",
    "readability_text_len_robust",
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


def _readability_extract_len(html: str) -> int:
    html = html.translate(str.maketrans('', '', ''.join(chr(i) for i in range(0x20))))
    if not html:
        return 0
    doc = Document(html)
    snippet = doc.summary(html_partial=True)
    text = BeautifulSoup(snippet, "lxml").get_text(" ", strip=True)
    return len(text)


def readability_text_len_robust(html: str) -> int:
    """Run Readability with staged fallbacks to avoid crashes on malformed HTML."""

    if not html:
        return 0

    def _attempt(payload: str) -> int:
        if not payload:
            return 0
        try:
            return _readability_extract_len(payload)
        except Exception:
            return 0

    length = _attempt(html)
    if length:
        return length

    fixed = minimal_text_fix(html)
    length = _attempt(fixed)
    if length:
        return length

    repaired = repair_markup(fixed)
    length = _attempt(repaired)
    if length:
        return length

    cleaned = clean_conservative(repaired)
    length = _attempt(cleaned)
    if length:
        return length

    try:
        soup = BeautifulSoup(repaired, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        try:
            tag.decompose()
        except Exception:
            continue

    candidates: list[int] = []
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
            text = node.get_text(" ", strip=True)
            if text:
                candidates.append(len(text))
        if candidates:
            break

    if not candidates:
        para_texts = []
        for node in soup.find_all("p")[:400]:
            txt = node.get_text(" ", strip=True)
            if txt:
                para_texts.append(len(txt))
        if para_texts:
            candidates.append(sum(para_texts))

    return max(candidates) if candidates else 0
