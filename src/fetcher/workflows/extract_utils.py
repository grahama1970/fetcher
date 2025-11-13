"""Utilities for content extraction/quality checks (trafilatura + readability)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup

try:  # Trafilatura is preferred for main-text extraction
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None  # type: ignore

try:  # readability-lxml fallback
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore


_PAYWALL_MARKERS = [
    "subscribe",
    "paywall",
    "sign in",
    "sign-in",
    "sign up",
    "sign-up",
    "log in",
    "login",
    "register to read",
    "members only",
    "premium",
    "unlock full",
    "enable javascript",
    "cookie consent",
]

_MIN_ARTICLE_CHARS = int(os.getenv("FETCHER_MIN_ARTICLE_CHARS", "1500"))
_MIN_WEAK_CHARS = int(os.getenv("FETCHER_MIN_WEAK_CHARS", "400"))
_MAX_LINK_DENSITY = float(os.getenv("FETCHER_MAX_LINK_DENSITY", "0.45"))
_MARKER_MIN_HITS = int(os.getenv("FETCHER_PAYWALL_MARKERS_MIN", "2"))
_ENABLE_POSTCHECK = os.getenv("FETCHER_ENABLE_POSTCHECK", "1") != "0"


@dataclass
class ContentAssessment:
    text: str
    text_len: int
    source: str
    link_density: float
    marker_hits: List[str]
    verdict: str
    score: float
    reasons: List[str]


def _run_trafilatura(html: str, url: Optional[str]) -> Optional[str]:
    if trafilatura is None:
        return None
    try:
        return trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_recall=True,
        )
    except Exception:  # pragma: no cover - trafilatura internal
        return None


def _run_readability(html: str) -> Optional[str]:
    if Document is None:
        return None
    try:
        doc = Document(html)
        summary = doc.summary(html_partial=True)
    except Exception:
        return None
    soup = BeautifulSoup(summary, "lxml")
    text = soup.get_text(" ", strip=True)
    return text or None


def _compute_link_density(html: str, extracted_text_len: int) -> float:
    if not html or extracted_text_len <= 0:
        return 0.0
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return 0.0
    total_link_chars = 0
    for link in soup.find_all("a"):
        total_link_chars += len(link.get_text(" ", strip=True))
    return min(1.0, total_link_chars / max(1, extracted_text_len))


def _marker_hits(text: str) -> List[str]:
    lowered = text.lower()
    hits = []
    for marker in _PAYWALL_MARKERS:
        if marker in lowered:
            hits.append(marker)
    return hits


def _assess_verdict(text_len: int, link_density: float, markers: List[str]) -> (str, float, List[str]):
    reasons: List[str] = []
    score = min(1.0, max(0.0, text_len / max(_MIN_ARTICLE_CHARS, 1)))
    marker_hits = len(markers)
    if text_len >= _MIN_ARTICLE_CHARS and marker_hits < _MARKER_MIN_HITS:
        return "ok", score, reasons
    if marker_hits >= _MARKER_MIN_HITS and text_len < _MIN_ARTICLE_CHARS:
        reasons.append("paywall_markers")
        return "paywall", score, reasons
    if link_density > _MAX_LINK_DENSITY and text_len < _MIN_ARTICLE_CHARS:
        reasons.append("high_link_density")
        return "link_hub", score, reasons
    if text_len < _MIN_WEAK_CHARS:
        reasons.append("too_short")
        return "thin", score, reasons
    reasons.append("weak_content")
    return "weak", score, reasons


def extract_content_features(text_or_html: str, content_type: str, url: Optional[str]) -> ContentAssessment:
    html = text_or_html or ""
    if not html.strip():
        return ContentAssessment("", 0, "none", 0.0, [], "thin", 0.0, ["empty_body"])

    is_html = "html" in (content_type or "").lower()
    extracted_text = html
    source = "raw"

    if is_html:
        extracted_text = _run_trafilatura(html, url)
        if extracted_text:
            source = "trafilatura"
        else:
            extracted_text = _run_readability(html)
            if extracted_text:
                source = "readability"
        if not extracted_text:
            soup = BeautifulSoup(html, "lxml")
            extracted_text = soup.get_text(" ", strip=True)
            source = "soup"
    else:
        extracted_text = html.strip()

    extracted_text = extracted_text or ""
    text_len = len(extracted_text)
    density = _compute_link_density(html, text_len) if is_html else 0.0
    markers = _marker_hits(extracted_text) if text_len else []
    verdict, score, reasons = _assess_verdict(text_len, density, markers)
    return ContentAssessment(extracted_text, text_len, source, density, markers, verdict, score, reasons)


def evaluate_result_content(result) -> None:
    """Annotate FetchResult with content verdict; blank text for junk."""

    html = result.text or ""
    assessment = extract_content_features(html, result.content_type or "", result.url)
    metadata = dict(result.metadata or {})
    metadata.update(
        {
            "content_verdict": assessment.verdict,
            "content_score": round(assessment.score, 3),
            "content_source": assessment.source,
            "content_text_len": assessment.text_len,
            "content_link_density": round(assessment.link_density, 3),
            "content_marker_hits": assessment.marker_hits,
            "content_reasons": assessment.reasons,
        }
    )

    if assessment.verdict != "ok":
        metadata["content_excerpt"] = html[:500]
        result.raw_bytes = None
    result.metadata = metadata


def verify_blob_content(result, blob_data: Optional[bytes]) -> None:
    """Optional post-write verification for persisted blobs."""

    if not _ENABLE_POSTCHECK or not blob_data:
        return
    metadata = dict(result.metadata or {})
    if metadata.get("content_verdict") != "ok":
        return
    try:
        text = blob_data.decode("utf-8", "ignore")
    except Exception:
        return
    reassessment = extract_content_features(text, result.content_type or "", result.url)
    if reassessment.verdict != "ok":
        metadata["content_verdict"] = "postcheck_failed"
        metadata.setdefault("content_reasons", []).append("postcheck_reject")
    metadata["content_text_len_post"] = reassessment.text_len
    result.metadata = metadata


__all__ = [
    "evaluate_result_content",
    "verify_blob_content",
]
