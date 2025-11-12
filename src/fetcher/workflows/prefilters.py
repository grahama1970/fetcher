from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup  # type: ignore


@dataclass(frozen=True)
class PrefilterDecision:
    keep: bool
    reason: Optional[str] = None
    section_pref_fallback: Optional[str] = None


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _load_domain_overrides() -> Dict[str, Dict[str, Any]]:
    """
    Load per-domain overrides for body_chars_min, paragraphs_min, heading_density_min.
    Env: SPARTA_PREFILTER_DOMAIN_OVERRIDES='{"host":{"body_chars_min":800,"paragraphs_min":4,"heading_density_min":0.02}}'
    """
    raw = os.getenv("SPARTA_PREFILTER_DOMAIN_OVERRIDES", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in data.items():
                if not isinstance(k, str) or not isinstance(v, dict):
                    continue
                out[k.lower()] = v
            return out
    except Exception:
        return {}
    return {}


def _defaults_for(url: str) -> Dict[str, Any]:
    """
    Resolve defaults with optional per-domain overrides.
    """
    # Baseline defaults for fan-out children (pre-Step07)
    base_chars_min = int(os.getenv("SPARTA_CHILD_BODY_CHARS_MIN", "600"))
    base_paras_min = int(os.getenv("SPARTA_CHILD_PARAGRAPHS_MIN", "3"))
    # Heading density threshold is optional; if set, filters listicle-like pages
    # density ~ (#headings)/(#paragraphs). Typical pages exceed ~0.02.
    hd_env = os.getenv("SPARTA_HEADING_DENSITY_MIN", "").strip()
    base_hd_min = float(hd_env) if hd_env else None
    host = _host(url)
    ovr = _load_domain_overrides().get(host, {})
    return {
        "body_chars_min": int(ovr.get("body_chars_min", base_chars_min)),
        "paragraphs_min": int(ovr.get("paragraphs_min", base_paras_min)),
        "heading_density_min": (float(ovr.get("heading_density_min")) if "heading_density_min" in ovr else base_hd_min),
    }


def _analyze_html(text: str) -> Dict[str, Any]:
    """
    Extract body text, paragraph count, heading count, and a simple 'section preference' fallback reason.
    """
    try:
        soup = BeautifulSoup(text, "lxml")
    except Exception:
        soup = None
    if not soup:
        body = (text or "").strip()
        paras = [p for p in (body.split("\n\n") if body else []) if p.strip()]
        return {
            "body": body,
            "body_len": len(body),
            "para_count": len(paras),
            "heading_count": 0,
            "heading_density": 0.0,
            "section_pref_fallback": "sections_missing" if not body else ("sections_too_small" if len(body) < 300 else None),
        }
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header", "aside"]):
        try:
            tag.decompose()
        except Exception:
            continue
    body_text = soup.get_text("\n")
    body = (body_text or "").strip()
    paras = [p for p in (body.split("\n\n") if body else []) if p.strip()]
    para_count = len(paras)
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    heading_count = len(headings)
    heading_density = (heading_count / max(1, para_count)) if para_count else 0.0
    # Section-preference fallback reason: prefer h2/h3; if missing entirely or text too small, signal why
    section_pref_fallback = None
    if heading_count == 0:
        section_pref_fallback = "sections_missing"
    elif len(body) < 300 or para_count < 2:
        section_pref_fallback = "sections_too_small"
    return {
        "body": body,
        "body_len": len(body),
        "para_count": para_count,
        "heading_count": heading_count,
        "heading_density": heading_density,
        "section_pref_fallback": section_pref_fallback,
    }


def evaluate_body_prefilter(*, text: str, content_type: str, url: str) -> PrefilterDecision:
    """
    Centralized prefilter evaluation for web_fetch hub fan-out and chunking pre-screens.
    Applies thresholds for:
      - minimum body character count,
      - minimum paragraph count,
      - optional minimum heading density (when configured).
    Adds a section-preference fallback reason for downstream telemetry.
    """
    ct = (content_type or "").lower()
    thresholds = _defaults_for(url)
    if ct.startswith("text/html"):
        info = _analyze_html(text or "")
        body_len = int(info["body_len"])
        para_count = int(info["para_count"])
        heading_density = float(info["heading_density"])
        hd_min = thresholds.get("heading_density_min", None)
        if body_len < int(thresholds["body_chars_min"]):
            return PrefilterDecision(keep=False, reason="low_body_text", section_pref_fallback=info.get("section_pref_fallback"))
        if para_count < int(thresholds["paragraphs_min"]):
            return PrefilterDecision(keep=False, reason="few_paragraphs", section_pref_fallback=info.get("section_pref_fallback"))
        if (hd_min is not None) and (heading_density < float(hd_min)):
            return PrefilterDecision(keep=False, reason="low_heading_density", section_pref_fallback=info.get("section_pref_fallback"))
        return PrefilterDecision(keep=True, reason=None, section_pref_fallback=info.get("section_pref_fallback"))
    # Non-HTML: evaluate by plain-text characteristics
    body = (text or "").strip()
    paras = [p for p in (body.split("\n\n") if body else []) if p.strip()]
    if len(body) < int(thresholds["body_chars_min"]):
        return PrefilterDecision(keep=False, reason="low_body_text", section_pref_fallback="sections_missing")
    if len(paras) < int(thresholds["paragraphs_min"]):
        return PrefilterDecision(keep=False, reason="few_paragraphs", section_pref_fallback="sections_too_small")
    return PrefilterDecision(keep=True, reason=None, section_pref_fallback=None)

