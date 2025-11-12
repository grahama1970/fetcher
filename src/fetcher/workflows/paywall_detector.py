"""Weighted paywall / unauthorized content detector."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional

from bs4 import BeautifulSoup

from .html_normalize import (
    minimal_text_fix,
    repair_markup,
    readability_text_len_robust,
)

VERSION = "1.3.0"

DEFAULT_WEIGHTS = {
    "unauthorized_status": 1.00,
    "payment_required": 0.90,
    "rate_limited": 0.40,
    "paywall_phrases": 0.60,
    "paywall_phrases_soft": 0.20,
    "vendor_markers": 0.50,
    "overlay_dom": 0.40,
    "readability_gap": 0.70,
    "short_body_stub": 0.20,
    "heavy_js": 0.30,
    "legalese_plus_subscribe": 0.30,
    "auth_header": 0.80,
    "paywall_header": 1.00,
    "noscript_paywall": 0.40,
    "meta_paywall": 0.40,
}

DEFAULT_CAPS = {
    "paywall_phrases": 1.00,
    "paywall_phrases_soft": 0.40,
    "vendor_markers": 1.00,
    "overlay_dom": 0.80,
}

DEFAULT_THRESHOLDS = {
    "likely": 1.00,
    "maybe": 0.50,
}

PAYWALL_STRONG_PHRASES = [
    r"\bsubscription required\b",
    r"\b(subscriber|members?) only\b",
    r"\bsign in to (?:continue|read)\b",
    r"\blog in to (?:continue|read)\b",
    r"\bthis article is for subscribers\b",
    r"\byou (?:have|used) [0-9]+ free articles\b",
    r"\bunlimited access\b",
    r"\bpaywall\b",
    r"\bnot available in your region\b",
]

PAYWALL_WEAK_PHRASES = [
    r"\bsubscribe\b",
    r"\bsubscription\b",
    r"\bplease enable javascript\b",
    r"\bthis is a potential security issue\b",
    r"\byou are viewing this page in an unauthorized frame\b",
]

VENDOR_PATTERNS = [
    r"\btinypass\b", r"\bpiano\b", r"\blaterpay\b", r"\bpressplus\b",
    r"\bzephr\b", r"\bcleeng\b", r"\bmemberful\b", r"\bblueconic\b",
    r"\b(paywall|metered)Wall\b",
]

LEGAL_TERMS = [
    r"\bterms of service\b",
    r"\bterms and conditions\b",
    r"\bcopyright\b",
    r"\bdmca\b",
    r"\ball rights reserved\b",
]

OVERLAY_HINTS = [
    r"\bpaywall\b", r"\boverlay\b", r"\bmodal\b", r"\bsignin\b",
    r"\bgate\b", r"\bwall\b",
]

META_HINTS = [
    r"\bmetered\b", r"\bpaywall\b", r"\bsubscription\b",
    r"\baccess(?:-)?control\b", r"\bmember(?:ship)?\b",
]

_RE_PAYWALL_STRONG = [re.compile(p, re.I) for p in PAYWALL_STRONG_PHRASES]
_RE_PAYWALL_WEAK = [re.compile(p, re.I) for p in PAYWALL_WEAK_PHRASES]
_RE_VENDOR = [re.compile(p, re.I) for p in VENDOR_PATTERNS]
_RE_LEGAL = [re.compile(p, re.I) for p in LEGAL_TERMS]
_RE_OVERLAY = [re.compile(p, re.I) for p in OVERLAY_HINTS]
_RE_META = [re.compile(p, re.I) for p in META_HINTS]


def _get_policy_attr(policy: Any, key: str, default):
    if policy is None:
        return default
    if isinstance(policy, Mapping):
        return policy.get(key, default)
    return getattr(policy, key, default)


def _merge_numbers(base: Mapping[str, float], override: Optional[Mapping[str, float]]) -> Dict[str, float]:
    return {**base, **(override or {})}


def _unique_hits(patterns: List[re.Pattern], text: str) -> List[str]:
    found = set()
    for pat in patterns:
        try:
            if pat.search(text):
                found.add(pat.pattern)
        except re.error:
            continue
    return sorted(found)


def _sanitize_html(html: str) -> str:
    try:
        return (html or "").replace("\x00", "")
    except Exception:
        return html or ""


def _extract_text(html: str) -> tuple[str, str, BeautifulSoup]:
    normalized = minimal_text_fix(_sanitize_html(html))
    repaired = repair_markup(normalized)
    soup = BeautifulSoup(repaired, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    noscripts = soup.find_all("noscript")
    noscript_text = " ".join(ns.get_text(" ", strip=True) for ns in noscripts) if noscripts else ""
    for ns in noscripts:
        ns.decompose()
    visible = soup.get_text(" ", strip=True)
    return visible, noscript_text, soup


def _readability_len(html: str) -> int:
    return readability_text_len_robust(_sanitize_html(html))


def _heavy_js_score(soup: BeautifulSoup) -> float:
    tags = soup.find_all(True)
    n_tags = max(1, len(tags))
    scripts = soup.find_all("script")
    n_scripts = len(scripts)
    externals = sum(1 for s in scripts if s.get("src"))
    density = n_scripts / n_tags
    inline_bytes = sum(len((s.string or "")) for s in scripts if not s.get("src"))
    if density >= 0.25 or externals >= 10 or inline_bytes >= 80_000:
        return 1.0
    if density >= 0.15 or externals >= 5 or inline_bytes >= 30_000:
        return 0.6
    if density >= 0.08 or externals >= 3 or inline_bytes >= 10_000:
        return 0.3
    return 0.0


def _detect_overlay_dom(soup: BeautifulSoup) -> List[str]:
    hits: List[str] = []
    for attr in ("class", "id"):
        for el in soup.find_all(True, attrs={attr: True}):
            val = el.get(attr)
            if isinstance(val, list):
                val = " ".join(val)
            val = val or ""
            if any(p.search(val) for p in _RE_OVERLAY):
                hits.append(f"{attr}:{val[:120]}")
                if len(hits) >= 10:
                    return hits
    for el in soup.find_all(True, attrs={"style": True}):
        style = (el.get("style") or "").lower()
        if ("position:fixed" in style or "position:absolute" in style) and (
            "top:0" in style or "left:0" in style
        ) and ("width:100%" in style or "height:100%" in style):
            hits.append(f"style:{style[:120]}")
            if len(hits) >= 10:
                break
    return hits


def _legalese_plus_subscribe(text: str) -> bool:
    has_legal = bool(_unique_hits(_RE_LEGAL, text))
    has_sub = bool(re.search(r"\bsubscribe\b", text, flags=re.I))
    return has_legal and has_sub


def _meta_paywall_hints(soup: BeautifulSoup) -> List[str]:
    hits = []
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").strip()
        content = (m.get("content") or "").strip()
        if not name and not content:
            continue
        target = f"{name}={content}"
        if any(p.search(target) for p in _RE_META):
            hits.append(target[:160])
            if len(hits) >= 10:
                break
    return hits


def _verdict_for(score: float, thresholds: Mapping[str, float]) -> str:
    likely = thresholds.get("likely", 1.0)
    maybe = thresholds.get("maybe", 0.5)
    if score >= likely:
        return "likely"
    if score >= maybe:
        return "maybe"
    return "unlikely"


def detect_paywall(
    url: str,
    status: Optional[int],
    html: str,
    headers: Optional[Mapping[str, str]] = None,
    policy: Any = None,
) -> Dict[str, Any]:
    """Return structured paywall detection scores for a fetched page."""

    headers = headers or {}
    weights = _merge_numbers(DEFAULT_WEIGHTS, _get_policy_attr(policy, "weights", None))
    caps = _merge_numbers(DEFAULT_CAPS, _get_policy_attr(policy, "caps", None))
    thresholds = _merge_numbers(DEFAULT_THRESHOLDS, _get_policy_attr(policy, "thresholds", None))
    extra_pay = [re.compile(p, re.I) for p in _get_policy_attr(policy, "paywall_hints", [])]
    extra_vendor = [re.compile(p, re.I) for p in _get_policy_attr(policy, "vendor_hints", [])]

    text, noscript_text, soup_full = _extract_text(html or "")

    detection: Dict[str, Any] = {
        "url": url,
        "status": status,
        "score": 0.0,
        "verdict": "unknown",
        "version": VERSION,
        "indicators": {},
        "score_breakdown": {},
        "meta": {},
    }

    if status in (401, 403, 451):
        detection["indicators"]["unauthorized_status"] = status
        detection["score"] += weights["unauthorized_status"]
        detection["score_breakdown"]["unauthorized_status"] = weights["unauthorized_status"]
        detection["verdict"] = _verdict_for(detection["score"], thresholds)
        return detection

    if status == 402:
        detection["indicators"]["payment_required"] = True
        detection["score"] += weights["payment_required"]
        detection["score_breakdown"]["payment_required"] = weights["payment_required"]

    if status == 429:
        detection["indicators"]["rate_limited"] = True
        detection["score"] += weights["rate_limited"]
        detection["score_breakdown"]["rate_limited"] = weights["rate_limited"]

    headers_lower = {k.lower(): v for k, v in headers.items()}
    auth_header = headers_lower.get("www-authenticate")
    if auth_header:
        detection["indicators"]["auth_header"] = auth_header[:160]
        detection["score"] += weights["auth_header"]
        detection["score_breakdown"]["auth_header"] = weights["auth_header"]
    paywall_header = headers_lower.get("x-paywall")
    if paywall_header and paywall_header.lower() in {"1", "true", "yes"}:
        detection["indicators"]["paywall_header"] = paywall_header
        detection["score"] += weights["paywall_header"]
        detection["score_breakdown"]["paywall_header"] = weights["paywall_header"]

    page_len = len(text)
    article_len = _readability_len(html)
    detection["meta"]["page_text_len"] = page_len
    detection["meta"]["article_len"] = article_len
    if page_len >= 1000:
        ratio = (article_len / page_len) if page_len else 0.0
        detection["meta"]["article_ratio"] = round(ratio, 4)
        if ratio <= 0.20 and article_len < 400:
            detection["indicators"]["readability_gap"] = True
            detection["score"] += weights["readability_gap"]
            detection["score_breakdown"]["readability_gap"] = weights["readability_gap"]
    if page_len < 400:
        detection["indicators"]["short_body_stub"] = True
        detection["score"] += weights["short_body_stub"]
        detection["score_breakdown"]["short_body_stub"] = weights["short_body_stub"]

    phrase_hits = _unique_hits(_RE_PAYWALL_STRONG + extra_pay, text)
    if phrase_hits:
        pts = min(len(phrase_hits) * weights["paywall_phrases"], caps.get("paywall_phrases", 1.0))
        detection["indicators"]["paywall_phrases"] = phrase_hits
        detection["score"] += pts
        detection["score_breakdown"]["paywall_phrases"] = round(pts, 3)

    weak_hits = _unique_hits(_RE_PAYWALL_WEAK, text)
    if weak_hits:
        pts = min(
            len(weak_hits) * weights.get("paywall_phrases_soft", 0.2),
            caps.get("paywall_phrases_soft", 0.4),
        )
        detection["indicators"]["paywall_phrases_weak"] = weak_hits
        detection["score"] += pts
        detection["score_breakdown"]["paywall_phrases_weak"] = round(pts, 3)

    vendor_hits = _unique_hits(_RE_VENDOR + extra_vendor, html or "")
    if vendor_hits:
        pts = min(len(vendor_hits) * weights["vendor_markers"], caps.get("vendor_markers", 1.0))
        detection["indicators"]["vendor_markers"] = vendor_hits
        detection["score"] += pts
        detection["score_breakdown"]["vendor_markers"] = round(pts, 3)

    overlay_hits = _detect_overlay_dom(soup_full)
    if overlay_hits:
        pts = min(weights["overlay_dom"], caps.get("overlay_dom", 0.8))
        detection["indicators"]["overlay_dom"] = overlay_hits
        detection["score"] += pts
        detection["score_breakdown"]["overlay_dom"] = round(pts, 3)

    js_score = _heavy_js_score(soup_full)
    if js_score > 0:
        pts = weights["heavy_js"] * js_score
        detection["indicators"]["heavy_js"] = js_score
        detection["score"] += pts
        detection["score_breakdown"]["heavy_js"] = round(pts, 3)

    if _legalese_plus_subscribe(text):
        detection["indicators"]["legalese_plus_subscribe"] = True
        detection["score"] += weights["legalese_plus_subscribe"]
        detection["score_breakdown"]["legalese_plus_subscribe"] = weights["legalese_plus_subscribe"]

    if noscript_text:
        ns_hits = _unique_hits(_RE_PAYWALL_STRONG + extra_pay, noscript_text)
        if ns_hits:
            detection["indicators"]["noscript_paywall"] = ns_hits
            detection["score"] += weights["noscript_paywall"]
            detection["score_breakdown"]["noscript_paywall"] = weights["noscript_paywall"]

    meta_hits = _meta_paywall_hints(soup_full)
    if meta_hits:
        detection["indicators"]["meta_paywall"] = meta_hits
        detection["score"] += weights["meta_paywall"]
        detection["score_breakdown"]["meta_paywall"] = weights["meta_paywall"]

    detection["verdict"] = _verdict_for(detection["score"], thresholds)
    return detection


__all__ = ["detect_paywall", "DEFAULT_WEIGHTS", "DEFAULT_CAPS", "DEFAULT_THRESHOLDS", "VERSION"]
