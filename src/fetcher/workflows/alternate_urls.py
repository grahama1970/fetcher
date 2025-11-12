from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # optional Brave Search integration
    from brave import BraveSearch  # type: ignore
except Exception:  # pragma: no cover
    BraveSearch = None  # type: ignore

# CHUTES/SciLLM: use acompletion (Bearer + JSON mode) consistently
try:
    from scillm import acompletion  # type: ignore
except Exception as e:  # pragma: no cover
    acompletion = None  # type: ignore


@dataclass
class FailedControl:
    control_id: str
    title: str
    failed_url: str
    worksheet: Optional[str]
    domain: Optional[str]
    category: Optional[str]
    definition: Optional[str]
    source_title: Optional[str] = None
    contexts: Optional[List[Dict[str, Any]]] = None


@dataclass
class AlternateSuggestion:
    control_id: str
    failed_url: str
    suggested_url: str
    source_title: Optional[str]
    summary: Optional[str]
    provider: str
    score: Optional[float]
    html_excerpt: Optional[str] = None


@dataclass
class AlternateResult:
    control: FailedControl
    suggestions: List[AlternateSuggestion]
    raw_response: str
    error: Optional[str] = None


def _require_scillm() -> None:
    if acompletion is None:
        raise RuntimeError("SciLLM not available; please install scillm and ensure CHUTES env is set")


def _chutes_env() -> Tuple[str, str, str]:
    api_base = os.environ.get("CHUTES_API_BASE")
    api_key = os.environ.get("CHUTES_API_KEY")
    model = os.environ.get("CHUTES_TEXT_MODEL")
    if not api_base or not api_key or not model:
        raise RuntimeError("Missing CHUTES env: set CHUTES_API_BASE, CHUTES_API_KEY, CHUTES_TEXT_MODEL")
    return api_base, api_key, model


def _build_prompt(control: FailedControl) -> Dict[str, Any]:
    payload = {
        "failed_url": control.failed_url,
        "control_id": control.control_id,
        "title": control.title,
        "worksheet": control.worksheet,
        "domain": control.domain,
        "category": control.category,
        "definition": control.definition,
        "source_title": control.source_title,
        "contexts": control.contexts or [],
    }
    return payload


def _query_brave(control: FailedControl, *, max_hits: int = 5) -> List[AlternateSuggestion]:
    if BraveSearch is None:
        return []
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return []
    try:
        client = BraveSearch(api_key)
    except Exception:
        return []
    title = control.title or control.source_title or control.control_id
    query = f"{title} {control.failed_url}".strip()
    try:
        resp = client.search(query, count=max_hits)
    except Exception:
        return []
    results = resp.get("web", {}).get("results", []) if isinstance(resp, dict) else []
    suggestions: List[AlternateSuggestion] = []
    for hit in results[:max_hits]:
        url = hit.get("url")
        if not url:
            continue
        suggestions.append(
            AlternateSuggestion(
                control_id=control.control_id,
                failed_url=control.failed_url,
                suggested_url=url,
                source_title=hit.get("title") or hit.get("name"),
                summary=hit.get("snippet") or hit.get("description"),
                provider="brave",
                score=None,
                html_excerpt=hit.get("snippet") or hit.get("description"),
            )
        )
    return suggestions


def _brave_enabled() -> bool:
    """Decide whether to use Brave fallback.

    Behavior:
    - If SPARTA_STEP06B_USE_BRAVE is set to 1/true/on → enabled
    - If set to 0/false/off → disabled
    - If unset or "auto" → enabled when BRAVE_API_KEY is present
    """
    val = os.getenv("SPARTA_STEP06B_USE_BRAVE")
    if val is None or val.strip().lower() == "auto":
        return bool(os.getenv("BRAVE_API_KEY"))
    v = val.strip().lower()
    return v in {"1", "true", "yes", "on"}


EXCLUDED_DOMAINS = {
    "sparta.aerospace.org",
    "wikipedia.org",
    "en.wikipedia.org",
    "wikimedia.org",
    "twitter.com",
    "x.com",
}


def _build_messages(control: FailedControl) -> List[Dict[str, Any]]:
    system_msg = (
        "You are assisting with space-security controls by locating accessible public content that matches a given document title. "
        "Use the structured context to precision-match the document and supply the best available HTML excerpt. "
        "Return JSON: {'alternates': [...], 'note': <optional string>}. "
        "Each alternate must include keys 'title', 'url', 'summary', 'html_excerpt' (provide at least two paragraphs of relevant HTML/plaintext suitable for knowledge chunking), and optional 'score'. "
        "Prefer authoritative primary sources such as NASA, ESA, GAO, CRS, FAA, CISA, vendor incident reports, or reputable open newsrooms. "
        "If the live page is gone, you may return an openly accessible archive such as the Internet Archive's Wayback cache when that preserves the same content. "
        "Do NOT return links that require login or subscription, and do NOT return SPARTA, Wikipedia, or derivative summaries. "
        "If no public source exists for the exact title, set alternates to an empty list and explain the limitation in 'note'."
    )
    user_payload = json.dumps(_build_prompt(control), ensure_ascii=False)
    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_payload},
    ]


async def _chutes_one(messages: List[Dict[str, Any]], *, api_base: str, api_key: str, model: str, timeout: float, max_tokens: int = 512, temperature: float = 0.2) -> Any:
    _require_scillm()
    resp = await acompletion(
        model=model,
        api_base=api_base,
        api_key=api_key,
        custom_llm_provider="openai_like",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        timeout=int(timeout),
    )
    # Return raw response dict-like for _parse_response compatibility
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    return resp


def _parse_response(raw: Any) -> Tuple[str, List[Dict[str, Any]], Optional[str], bool]:
    if raw is None:
        return "", [], None
    if hasattr(raw, "model_dump"):
        raw_dict = raw.model_dump()
    elif hasattr(raw, "to_dict"):
        raw_dict = raw.to_dict()
    elif isinstance(raw, dict):
        raw_dict = raw
    else:
        raw_dict = json.loads(json.dumps(raw))
    content = (
        raw_dict.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    if not content:
        return "", [], None, False
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = None
        # Attempt to extract fenced code block
        if "```" in content:
            parts = content.split("```")
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("json"):
                    candidate = part[4:].strip()
                else:
                    candidate = part
                if not candidate.startswith("{"):
                    continue
                try:
                    data = json.loads(candidate)
                    break
                except json.JSONDecodeError:
                    continue
        if data is None:
            # Fallback: attempt to parse substring from first '{' to last '}'
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = content[start : end + 1]
                try:
                    data = json.loads(snippet)
                except json.JSONDecodeError:
                    data = None
        if data is None:
            return content, [], None, False
    note = data.get("note") if isinstance(data, dict) else None
    alternates = data.get("alternates") if isinstance(data, dict) else None
    if not isinstance(alternates, list):
        return content, [], note, False
    cleaned: List[Dict[str, Any]] = []
    for item in alternates:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        summary = (item.get("summary") or "").strip() or None
        html_excerpt = (item.get("html_excerpt") or item.get("html") or item.get("content") or "").strip() or None
        if not html_excerpt and summary:
            html_excerpt = summary
        cleaned.append(
            {
                "url": url,
                "title": (item.get("title") or "").strip() or None,
                "summary": summary,
                "score": item.get("score"),
                "html_excerpt": html_excerpt,
            }
        )
    return content, cleaned, note, True


def _normalize_title(title: Optional[str]) -> str:
    if not title:
        return ""
    return re.sub(r"[^a-z0-9]+", "", title.lower())


_PAYWALL_HINTS = (
    "subscription",
    "subscribe",
    "paywall",
    "login",
    "sign in",
    "sign-in",
    "account",
    "purchase",
    "copyright",
    "proprietary",
    "members only",
)


def _looks_paywalled(url: str, summary: Optional[str]) -> bool:
    lower_url = url.lower()
    if any(token in lower_url for token in ("login", "signin", "subscribe", "paywall", "account")):
        return True
    if summary:
        lowered = summary.lower()
        if any(hint in lowered for hint in _PAYWALL_HINTS):
            return True
    return False


def generate_alternate_urls(
    controls: Iterable[FailedControl],
    *,
    concurrency: int = 3,
    max_retries: int = 3,
    retry_after: float = 2.0,
    timeout: float = 45.0,
) -> Tuple[List[AlternateResult], List[AlternateResult]]:
    control_list = list(controls)
    if not control_list:
        return [], []

    async def runner() -> List[Tuple[FailedControl, Any, Optional[str]]]:
        api_base, api_key, model = _chutes_env()
        messages_list = [_build_messages(control) for control in control_list]
        total = len(messages_list)
        effective_concurrency = min(concurrency, total) or 1

        pending = list(range(total))
        results: List[Optional[Any]] = [None] * total
        errors: List[Optional[str]] = [None] * total
        attempt = 0

        while pending and attempt <= max_retries:
            current_idx = list(pending)
            sem = asyncio.Semaphore(min(effective_concurrency, len(current_idx)))
            async def _task(i: int):
                async with sem:
                    try:
                        return await _chutes_one(messages_list[i], api_base=api_base, api_key=api_key, model=model, timeout=timeout)
                    except Exception as e:
                        return e
            raw_results = await asyncio.gather(*(_task(i) for i in current_idx), return_exceptions=False)

            next_pending: List[int] = []
            for idx, raw in zip(pending, raw_results):
                error: Optional[str] = None
                if isinstance(raw, Exception):
                    error = str(raw)
                elif isinstance(raw, dict):
                    # No router meta in CHUTES path; treat as OK
                    pass

                if error:
                    results[idx] = raw
                    errors[idx] = error
                    if attempt < max_retries:
                        next_pending.append(idx)
                else:
                    # Strict JSON parse; schedule one retry on parse_error
                    raw_text, suggestions, note, ok = _parse_response(raw)
                    if not ok and attempt < max_retries:
                        results[idx] = raw
                        errors[idx] = "parse_error"
                        next_pending.append(idx)
                    else:
                        results[idx] = raw
                        errors[idx] = None

            pending = next_pending
            attempt += 1
            if pending and attempt <= max_retries and retry_after:
                await asyncio.sleep(retry_after)

        output: List[Tuple[FailedControl, Any, Optional[str]]] = []
        for control, raw, error in zip(control_list, results, errors):
            output.append((control, raw, error))
        return output

    # Run on a dedicated loop to avoid cross-loop worker issues
    try:
        import asyncio
        _ALT_LOOP = getattr(sys.modules[__name__], "_ALT_LOOP", None)
    except Exception:
        _ALT_LOOP = None
    if _ALT_LOOP is None or _ALT_LOOP.is_closed():
        _ALT_LOOP = asyncio.new_event_loop()
        setattr(sys.modules[__name__], "_ALT_LOOP", _ALT_LOOP)
    try:
        responses = _ALT_LOOP.run_until_complete(runner())
    except Exception:
        try:
            _ALT_LOOP.close()
        except Exception:
            pass
        _ALT_LOOP = asyncio.new_event_loop()
        setattr(sys.modules[__name__], "_ALT_LOOP", _ALT_LOOP)
        responses = _ALT_LOOP.run_until_complete(runner())
    successes: List[AlternateResult] = []
    failures: List[AlternateResult] = []

    for control, raw, error in responses:
        if error:
            failures.append(
                AlternateResult(
                    control=control,
                    suggestions=[],
                    raw_response=json.dumps(raw, ensure_ascii=False) if raw is not None else "",
                    error=error,
                )
            )
            continue
        raw_text, suggestions, note, ok = _parse_response(raw)
        normalized_target = _normalize_title(control.source_title or control.title)
        strict_match = bool(control.source_title)
        filtered: List[AlternateSuggestion] = []
        for item in suggestions:
            candidate_title = item.get("title") or ""
            candidate_norm = _normalize_title(candidate_title)
            if strict_match and candidate_norm != normalized_target:
                continue
            url = item["url"]
            summary = item.get("summary")
            html_excerpt = item.get("html_excerpt")
            if not html_excerpt:
                continue
            if _looks_paywalled(url, summary):
                continue
            domain = urlparse(url).netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
            if domain in EXCLUDED_DOMAINS:
                continue
            filtered.append(
                AlternateSuggestion(
                    control_id=control.control_id,
                    failed_url=control.failed_url,
                    suggested_url=url,
                    source_title=candidate_title or None,
                    summary=summary,
                    provider="chutes",
                    score=item.get("score"),
                    html_excerpt=html_excerpt,
                )
            )

        if filtered:
            successes.append(
                AlternateResult(control=control, suggestions=filtered, raw_response=raw_text)
            )
        else:
            brave_enabled = _brave_enabled()
            brave_suggestions = _query_brave(control) if brave_enabled else []
            if brave_suggestions:
                successes.append(AlternateResult(control=control, suggestions=brave_suggestions, raw_response=raw_text))
            else:
                note = note or "No public alternate located for exact title or accessible content"
                failures.append(
                    AlternateResult(
                        control=control,
                        suggestions=[],
                        raw_response=raw_text,
                        error=note,
                    )
                )

    return successes, failures
