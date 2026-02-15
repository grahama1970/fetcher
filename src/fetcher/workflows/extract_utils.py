"""Utilities for content extraction/quality checks (trafilatura + readability)."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup

try:  # Trafilatura is preferred for main-text extraction
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None  # type: ignore

try:  # readability-lxml fallback
    from readability import Document  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    Document = None  # type: ignore

from .html_normalize import minimal_text_fix, readability_extract_text_robust


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
_DEPRECATION_KEYWORDS = ("deprecation warning", "deprecation notice", "deprecated")
_DEPRECATION_PATTERN = re.compile(r"deprecat", re.IGNORECASE)
_DEFINITION_TOKENS = (
    "definition:",
    "definitions:",
    "definition(s):",
    "definition",
    "definitions",
)

_LINK_HEAVY_DEFAULTS = {
    "attack.mitre.org",
    "mitre.org",
    "d3fend.mitre.org",
    "nist.gov",
    "csrc.nist.gov",
    "sparta.aerospace.org",
    "aerospace.org",
}


def _load_link_heavy_domains() -> List[str]:
    raw = os.getenv("FETCHER_LINK_HEAVY_DOMAINS", "")
    custom = [token.strip() for token in raw.split(",") if token.strip()]
    domains = set(_LINK_HEAVY_DEFAULTS)
    for token in custom:
        domains.add(token.lower().lstrip("."))
    return sorted(domains)


_LINK_HEAVY_DOMAINS = _load_link_heavy_domains()

_SOFT_404_PATTERNS = (
    "page you were looking for can't be found",
    "page you were looking for cannot be found",
    "page not found",
    "file not found",
    "doesn't exist",
    "does not exist",
    "sorry, the page",
    "404",
    "access denied",
    "errors.edgesuite.net",
)


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
    structured_definition: Optional[str] = None


def _run_trafilatura(html: str, url: Optional[str]) -> Optional[str]:
    if trafilatura is None:
        return None
    try:
        return trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        )
    except Exception:  # pragma: no cover - trafilatura internal
        return None


def _run_readability(html: str) -> Optional[str]:
    if Document is None:
        return None
    try:
        text = readability_extract_text_robust(html)
        return text or None
    except Exception:
        return None


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


def _detect_structured_definition(html: str) -> Optional[str]:
    if not html:
        return None
    lowered = html.lower()
    if "definition" not in lowered and "glossary" not in lowered:
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return None

    def _clean_snippet(node) -> Optional[str]:
        if not node:
            return None
        text = node.get_text(" ", strip=True)
        if text and 40 <= len(text) <= 1200:
            return text
        return None

    for tag in soup.find_all(["strong", "h1", "h2", "h3", "h4", "h5", "dt", "th", "p", "span", "div"]):
        text = tag.get_text(" ", strip=True)
        if not text:
            continue
        lowered_text = text.lower()
        if any(token in lowered_text for token in _DEFINITION_TOKENS):
            candidate = _clean_snippet(tag.parent)
            if candidate:
                return candidate[:800]
            candidate = _clean_snippet(tag.next_sibling)
            if candidate:
                return candidate[:800]
            return text

    dl_tag = soup.find("dl")
    if dl_tag:
        cleaned = _clean_snippet(dl_tag)
        if cleaned:
            return cleaned[:800]
    return None


def _detect_deprecation_warning(html: str) -> Optional[str]:
    if not html or not _DEPRECATION_PATTERN.search(html):
        return None
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return "deprecation warning"

    match = soup.find(string=_DEPRECATION_PATTERN)
    if not match:
        return "deprecation warning"

    def _clean_text(node) -> Optional[str]:
        if not node:
            return None
        text = node.get_text(" ", strip=True)
        if text and len(text) <= 800 and _DEPRECATION_PATTERN.search(text):
            return text
        return None

    parent = match.parent
    while parent and parent.name in {"script", "style"}:
        parent = parent.parent
    best_text: Optional[str] = None
    while parent:
        cleaned = _clean_text(parent)
        if cleaned:
            best_text = cleaned
        classes = []
        try:
            classes = [token.lower() for token in parent.get("class", [])]
        except Exception:
            classes = []
        if classes and any(token in {"card", "danger-card", "alert", "warning", "deprecated", "notice"} for token in classes):
            snippet = parent.get_text(" ", strip=True)
            if snippet:
                return snippet[:800]
        parent = parent.parent

    direct_text = match.strip()
    return (best_text or direct_text) or "deprecation warning"


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


def matches_link_heavy_domain(url: Optional[str]) -> bool:
    if not url:
        return False
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    host = host.lower().rstrip(".")
    if not host:
        return False
    for entry in _LINK_HEAVY_DOMAINS:
        token = entry.strip().lower().lstrip(".")
        if not token:
            continue
        if host == token or host.endswith(f".{token}"):
            return True
    return False


def _matches_link_heavy_domain(url: Optional[str]) -> bool:
    # Backwards compatibility alias for callers that import the old helper.
    return matches_link_heavy_domain(url)


def extract_content_features(text_or_html: str, content_type: str, url: Optional[str]) -> ContentAssessment:
    html = text_or_html or ""
    if not html.strip():
        return ContentAssessment("", 0, "none", 0.0, [], "thin", 0.0, ["empty_body"])

    # Normalize the body early for more robust extraction across weird encodings,
    # mojibake, and zero-width/control noise.
    html = minimal_text_fix(html)

    lowered_ct = (content_type or "").lower()
    is_html = "html" in lowered_ct
    is_pdf = "pdf" in lowered_ct
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
    # Relax paywall-style marker handling for PDFs: long-form reports frequently
    # mention generic commerce/account terminology without being access-gated.
    # For non-HTML content (notably application/pdf), ignore marker hits so that
    # PDF extraction success is not downgraded to "weak" or "paywall" purely
    # based on incidental token matches.
    if is_pdf:
        markers = []
    verdict, score, reasons = _assess_verdict(text_len, density, markers)
    structured_definition = _detect_structured_definition(html)
    if structured_definition and verdict in {"thin", "link_hub", "weak"}:
        verdict = "ok"
        reasons = [reason for reason in reasons if reason not in {"too_short", "high_link_density", "weak_content"}]
        reasons.append("structured_definition")
    parsed_host = ""
    parsed_path = ""
    if url:
        try:
            parsed = urlparse(url)
            parsed_host = (parsed.hostname or "").lower()
            parsed_path = parsed.path or ""
        except Exception:
            parsed_host = ""
            parsed_path = ""

    if verdict in {"link_hub", "weak"} and _matches_link_heavy_domain(url):
        verdict = "ok"
        reasons = [reason for reason in reasons if reason not in {"weak_content", "low_body_text"}]
        reasons.append("link_heavy_allow")

    if parsed_host.endswith("capec.mitre.org") and "/community/" in parsed_path:
        lowered_path = parsed_path.lower()
        if any(token in lowered_path for token in ("registration", "signup", "sign-up", "login")):
            verdict = "paywall"
            if "registration_required" not in reasons:
                reasons.append("registration_required")
        elif verdict in {"link_hub", "thin", "weak"}:
            if "community_portal" not in reasons:
                reasons.append("community_portal")

    return ContentAssessment(
        extracted_text,
        text_len,
        source,
        density,
        markers,
        verdict,
        score,
        reasons,
        structured_definition=structured_definition,
    )


def evaluate_result_content(result) -> None:
    """Annotate FetchResult with content verdict; blank text for junk."""

    metadata = dict(result.metadata or {})
    status = getattr(result, "status", None)

    # Password-protected PDFs are a deterministic skip unless we successfully
    # recovered the password upstream. When recovered, proceed with normal
    # content evaluation so text can be chunked.
    if metadata.get("pdf_password_protected") and not metadata.get("pdf_password_recovered"):
        metadata.update(
            {
                "content_verdict": "password_protected",
                "content_score": 0.0,
                "content_source": "pdf_meta",
                "content_text_len": 0,
                "content_link_density": 0.0,
                "content_marker_hits": [],
                "content_reasons": ["pdf_password_protected"],
            }
        )
        result.text = ""
        result.raw_bytes = None
        result.metadata = metadata
        return

    html = result.text or ""
    assessment = extract_content_features(html, result.content_type or "", result.url)
    verdict = assessment.verdict
    reasons = list(assessment.reasons)

    # Detect soft-404 placeholder PDFs (short bodies with classic "page not found" phrases)
    lowered_text = (assessment.text or "").lower()
    is_pdf = "pdf" in (result.content_type or "").lower() or bool(metadata.get("pdf_text_extracted"))
    if is_pdf and assessment.text_len < max(_MIN_WEAK_CHARS, 1200):
        if any(token in lowered_text for token in _SOFT_404_PATTERNS):
            verdict = "missing_file"
            reasons = [r for r in reasons if r != "too_short"]
            reasons.append("pdf_soft_404")
            metadata["soft_404_detected"] = True

    # PDF URL that returned non-PDF content or an empty body (soft 404 / HTML placeholder)
    url_is_pdf = (result.url or "").lower().endswith(".pdf")
    ct_is_pdf = "pdf" in (result.content_type or "").lower()
    pdf_text_missing = not metadata.get("pdf_text_extracted")
    if url_is_pdf and (not ct_is_pdf) and pdf_text_missing:
        verdict = "missing_file"
        reasons = [r for r in reasons if r != "too_short"]
        if "pdf_fetch_mismatch" not in reasons:
            reasons.append("pdf_fetch_mismatch")
        metadata["soft_404_detected"] = metadata.get("soft_404_detected") or True
    if url_is_pdf and status and status in {404, 410} and pdf_text_missing:
        verdict = "missing_file"
        if "pdf_status_error" not in reasons:
            reasons.append("pdf_status_error")

    lowered_ct = (result.content_type or "").lower()
    is_pdf = "pdf" in lowered_ct or bool(metadata.get("pdf_text_extracted"))
    is_youtube = getattr(result, "method", "") == "youtube-skill"

    # If we successfully extracted substantial text from a PDF, treat generic
    # paywall/weak-content signals as soft warnings instead of hard failures.
    # This prevents publicly available research PDFs from being zeroed out
    # downstream when they incidentally contain subscription language.
    if is_pdf and assessment.text_len >= _MIN_ARTICLE_CHARS and verdict in {"paywall", "weak"}:
        metadata["content_verdict_original"] = verdict
        if reasons:
            metadata["content_reasons_original"] = reasons
        verdict = "ok"
        reasons = [r for r in reasons if r not in {"too_short", "high_link_density", "weak_content", "paywall_markers"}]
        reasons.append("pdf_long_text_override")

    # YouTube transcripts often fall into 'weak' (short) or 'thin' (very short)
    # categories but are still valid and usable.
    if is_youtube and verdict in {"weak", "thin"} and assessment.text_len > 0:
        metadata["content_verdict_original"] = verdict
        if reasons:
            metadata["content_reasons_original"] = reasons
        verdict = "ok"
        reasons = [r for r in reasons if r not in {"too_short", "weak_content"}]
        reasons.append("youtube_transcript_preserve")

    warning_text = _detect_deprecation_warning(html)
    metadata.update(
        {
            "content_verdict": verdict,
            "content_score": round(assessment.score, 3),
            "content_source": assessment.source,
            "content_text_len": assessment.text_len,
            "content_link_density": round(assessment.link_density, 3),
            "content_marker_hits": assessment.marker_hits,
            "content_reasons": reasons,
        }
    )

    # Make it trivial for downstream agents/humans to decide whether a URL is
    # usable without re-implementing fetcher heuristics.
    usable = bool((status == 200) and (verdict == "ok"))
    metadata["usable"] = usable
    metadata["is_junk"] = not usable
    if not usable:
        if status != 200:
            metadata["junk_reason"] = f"http_status_{status}"
        else:
            metadata["junk_reason"] = f"content_verdict_{verdict}"
    else:
        metadata["junk_reason"] = None
    if warning_text:
        metadata["deprecation_warning"] = True
        metadata["deprecation_warning_text"] = warning_text[:1000]
    if assessment.structured_definition:
        metadata["structured_definition"] = True
        metadata["structured_definition_text"] = assessment.structured_definition

    if verdict != "ok":
        metadata["content_excerpt"] = html[:500]
        result.raw_bytes = None
        if verdict == "paywall":
            metadata["paywall_stub"] = True
        else:
            result.text = ""
    else:
        # Replace raw HTML with extracted clean text
        result.text = assessment.text
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
    # Postcheck is intended to catch corruption or unexpected decode/extraction
    # problems after writing bytes to disk. Keep the primary verdict stable for
    # downstream behavior, but record postcheck diagnostics explicitly.
    metadata["postcheck_verdict"] = reassessment.verdict
    metadata["postcheck_score"] = round(reassessment.score, 3)
    metadata["postcheck_reasons"] = reassessment.reasons
    metadata["content_text_len_post"] = reassessment.text_len
    if reassessment.verdict != "ok":
        metadata["postcheck_failed"] = True
    result.metadata = metadata


__all__ = [
    "evaluate_result_content",
    "verify_blob_content",
]
