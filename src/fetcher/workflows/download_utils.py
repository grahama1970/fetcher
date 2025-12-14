"""Helper utilities for externalizing bodies, rolling windows, and paywall metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import mimetypes
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup  # type: ignore

try:  # spaCy is optional; fall back to regex segmentation when missing
    import spacy  # type: ignore
except Exception:  # pragma: no cover - spaCy optional
    spacy = None  # type: ignore

from .paywall_detector import detect_paywall
from .extract_utils import extract_content_features, verify_blob_content
from .fetcher_utils import is_safe_domain as _is_safe_domain
from .web_fetch import FetchResult
from ..core.keys import K_TEXT_PATH

try:  # trafilatura is optional, but installed in the default fetcher env
    import trafilatura  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    trafilatura = None  # type: ignore

_SPACY_SENTENCIZER = None

_FIT_MARKDOWN_RULE_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def _load_fit_markdown_rules(overrides_path: Optional[Path]) -> List[Dict[str, Any]]:
    if overrides_path is None:
        return []
    try:
        key = str(overrides_path.resolve())
    except Exception:
        key = str(overrides_path)
    if key in _FIT_MARKDOWN_RULE_CACHE:
        return _FIT_MARKDOWN_RULE_CACHE[key]
    rules: List[Dict[str, Any]] = []
    try:
        if overrides_path.exists():
            data = json.loads(overrides_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    if item.get("fit_markdown_selectors"):
                        rules.append(item)
    except Exception:
        rules = []
    _FIT_MARKDOWN_RULE_CACHE[key] = rules
    return rules


def _pick_fit_markdown_selectors(url: str, overrides_path: Optional[Path]) -> List[str]:
    rules = _load_fit_markdown_rules(overrides_path)
    if not rules or not url:
        return []
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
    except Exception:
        return []

    candidates: List[Tuple[int, List[str]]] = []
    for rule in rules:
        domain = str(rule.get("domain") or "").strip().lower()
        if domain and domain != host:
            continue
        path_prefix = str(rule.get("path_prefix") or "").strip()
        if path_prefix and not path.startswith(path_prefix):
            continue
        substr = str(rule.get("substring") or "").strip()
        if substr and substr not in url:
            continue
        selectors = rule.get("fit_markdown_selectors") or []
        if not isinstance(selectors, list):
            continue
        cleaned = [str(s).strip() for s in selectors if str(s).strip()]
        if not cleaned:
            continue
        candidates.append((len(path_prefix), cleaned))
    if not candidates:
        return []
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]


def _select_html_by_css(html: str, selectors: List[str]) -> str:
    if not html or not selectors:
        return html
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return html
    picked = []
    seen = set()
    for sel in selectors:
        try:
            nodes = soup.select(sel)
        except Exception:
            continue
        for node in nodes[:8]:
            key = getattr(node, "name", None), id(node)
            if key in seen:
                continue
            seen.add(key)
            picked.append(str(node))
        if picked:
            break
    if not picked:
        return html
    return "<html><body>\n" + "\n".join(picked) + "\n</body></html>"


def _prune_markdown_to_first_h1(markdown: str) -> str:
    """Best-effort pruning for LLM-friendly markdown.

    Many markdown generators include stray UI tokens (e.g., `Esc`) before the
    first real heading. Trim everything before the first top-level heading.
    """

    if not markdown:
        return ""
    lines = markdown.splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.startswith("# "):
            start = idx
            break
    if start is None:
        return markdown.strip()
    return "\n".join(lines[start:]).strip()


def _get_spacy_sentencizer():
    global _SPACY_SENTENCIZER
    if spacy is None:
        return None
    if _SPACY_SENTENCIZER is None:
        try:
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
            _SPACY_SENTENCIZER = nlp
        except Exception:
            _SPACY_SENTENCIZER = None
    return _SPACY_SENTENCIZER


def maybe_externalize_text(
    results: Iterable[FetchResult],
    run_artifacts_dir: Path,
    max_inline_bytes: Optional[int],
) -> None:
    if max_inline_bytes is None or max_inline_bytes < 0:
        return

    text_root = run_artifacts_dir / "text_blobs"
    text_root.mkdir(parents=True, exist_ok=True)
    shared_root_env = os.getenv("FETCHER_TEXT_CACHE_DIR", "").strip()
    shared_root: Optional[Path] = None
    if shared_root_env:
        shared_root = Path(shared_root_env).expanduser()
        shared_root.mkdir(parents=True, exist_ok=True)

    for result in results:
        text = result.text or ""
        if not text:
            continue
        text_bytes = text.encode("utf-8")
        if max_inline_bytes > 0 and len(text_bytes) <= max_inline_bytes:
            continue

        sha = hashlib.sha256(text_bytes).hexdigest()
        shared_path: Optional[Path] = None
        if shared_root is not None:
            shared_path = shared_root / f"{sha}.txt"
            if not shared_path.exists():
                shared_path.write_text(text, encoding="utf-8")
        blob_path = shared_path or (text_root / f"{sha}.txt")
        if shared_path is None:
            blob_path.write_text(text, encoding="utf-8")
        else:
            run_copy = text_root / f"{sha}.txt"
            if not run_copy.exists():
                try:
                    os.symlink(shared_path, run_copy)
                except OSError:
                    if not run_copy.exists():
                        shutil.copy2(shared_path, run_copy)
        resolved_path = shared_path or blob_path

        metadata = dict(result.metadata or {})
        metadata[K_TEXT_PATH] = str(resolved_path)
        metadata["text_externalized"] = True
        metadata["text_inline_bytes"] = len(text_bytes)
        metadata["text_sha256"] = sha
        metadata["file_path"] = str(resolved_path)
        if shared_path is not None:
            metadata["text_cache_path"] = str(shared_path)
        metadata["text_length_chars"] = len(text)
        metadata["text_inline_missing"] = True

        result.metadata = metadata
        result.text = ""


def _guess_blob_extension(url: str, content_type: str) -> str:
    try:
        path_suffix = Path(urlparse(url).path or "").suffix
    except Exception:
        path_suffix = ""
    if path_suffix and len(path_suffix) <= 10:
        return path_suffix
    ext = mimetypes.guess_extension(content_type or "", strict=False)
    return ext or ".bin"


def _persist_blob_for_result(result: FetchResult, download_dir: Path) -> Optional[Path]:
    data = result.raw_bytes
    source = "raw"
    if data is None and result.text:
        data = result.text.encode("utf-8")
        source = "text"
    if data is None:
        return None

    sha = hashlib.sha256(data).hexdigest()
    ext = _guess_blob_extension(result.url, result.content_type)
    blob_path = download_dir / f"{sha}{ext}"
    if not blob_path.exists():
        blob_path.write_bytes(data)
        verify_blob_content(result, data)

    metadata = dict(result.metadata or {})
    metadata["blob_path"] = str(blob_path)
    metadata["blob_sha256"] = sha
    metadata["blob_size"] = len(data)
    metadata["blob_content_type"] = result.content_type
    metadata["blob_source"] = source
    metadata["blob_filename"] = blob_path.name
    metadata.setdefault("file_path", str(blob_path))
    result.metadata = metadata
    return blob_path


def _regex_sentence_split(text: str) -> List[Tuple[str, int, int]]:
    sentences: List[Tuple[str, int, int]] = []
    pattern = re.compile(r"([^.!?]+[.!?])", re.DOTALL)
    for match in pattern.finditer(text):
        segment = match.group(0)
        start = match.start()
        end = match.end()
        cleaned = segment.strip()
        if not cleaned:
            continue
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment) - len(segment.rstrip())
        sentences.append((cleaned, start + leading, end - trailing))
    if not sentences and text.strip():
        cleaned = text.strip()
        start = text.find(cleaned)
        sentences.append((cleaned, start, start + len(cleaned)))
    return sentences


def _split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    text = text or ""
    if not text.strip():
        return []

    nlp = _get_spacy_sentencizer()
    if nlp is not None:
        try:
            doc = nlp(text)
            sentences: List[Tuple[str, int, int]] = []
            for sent in doc.sents:
                raw = sent.text
                if not raw or raw.isspace():
                    continue
                leading = len(raw) - len(raw.lstrip())
                trailing = len(raw) - len(raw.rstrip())
                cleaned = raw.strip()
                if not cleaned:
                    continue
                start = sent.start_char + leading
                end = sent.end_char - trailing
                sentences.append((cleaned, start, end))
            if sentences:
                return sentences
        except Exception:
            pass

    return _regex_sentence_split(text)


def build_rolling_windows(
    text: str,
    window_size: int,
    window_step: int,
    max_windows: int,
) -> List[Dict[str, Any]]:
    if window_size <= 0 or window_step <= 0 or not text:
        return []

    sentences = _split_sentences_with_offsets(text)
    if not sentences:
        sentences = [(text.strip(), 0, len(text.strip()))]

    windows: List[Dict[str, Any]] = []
    total_sentences = len(sentences)
    start_idx = 0
    window_index = 0

    while start_idx < total_sentences:
        window_start_char = sentences[start_idx][1]
        end_idx = start_idx

        while end_idx + 1 < total_sentences:
            candidate_end = sentences[end_idx + 1][2]
            if candidate_end - window_start_char > window_size and end_idx >= start_idx:
                break
            end_idx += 1

        window_start = sentences[start_idx][1]
        window_end = sentences[end_idx][2]
        if window_end <= window_start:
            window_end = min(len(text), window_start + window_size)
        window_text = text[window_start:window_end].strip()
        if window_text:
            windows.append(
                {
                    "index": window_index,
                    "start": window_start,
                    "end": window_end,
                    "text": window_text,
                }
            )
            window_index += 1
        if max_windows > 0 and len(windows) >= max_windows:
            break

        target_char = window_start_char + window_step
        if target_char <= window_start_char:
            target_char = window_start_char + 1

        next_idx = start_idx
        while next_idx < total_sentences and sentences[next_idx][1] < target_char:
            next_idx += 1
        if next_idx == start_idx:
            next_idx += 1
        start_idx = next_idx

    return windows


def _attach_rolling_windows(
    result: FetchResult,
    window_dir: Path,
    window_size: int,
    window_step: int,
    max_windows: int,
) -> None:
    text = result.text or ""
    if not text and result.raw_bytes:
        try:
            text = result.raw_bytes.decode("utf-8", "ignore")
        except Exception:
            text = ""
    if not text:
        metadata = result.metadata or {}
        text_path = metadata.get(K_TEXT_PATH)
        if text_path:
            try:
                text = Path(text_path).read_text(encoding="utf-8")
            except Exception:
                text = ""
    if not text:
        return

    windows = build_rolling_windows(text, window_size, window_step, max_windows)
    if not windows:
        return

    metadata = dict(result.metadata or {})
    blob_sha = metadata.get("blob_sha256")
    if not blob_sha:
        blob_sha = hashlib.sha256((result.url or "").encode("utf-8")).hexdigest()
    window_dir.mkdir(parents=True, exist_ok=True)
    window_path = window_dir / f"{blob_sha}.jsonl"
    with window_path.open("w", encoding="utf-8") as fh:
        for window in windows:
            fh.write(json.dumps(window, ensure_ascii=False) + "\n")

    metadata["rolling_windows_path"] = str(window_path)
    metadata["rolling_windows_count"] = len(windows)
    metadata["rolling_window_size"] = window_size
    metadata["rolling_window_step"] = window_step
    if max_windows > 0:
        metadata["rolling_window_max_windows"] = max_windows
    result.metadata = metadata


def apply_download_mode(
    results: Iterable[FetchResult],
    run_artifacts_dir: Path,
    mode: str,
    window_size: int,
    window_step: int,
    max_windows: int,
) -> None:
    if mode == "text":
        for result in results:
            result.raw_bytes = None
        return

    run_artifacts_dir.mkdir(parents=True, exist_ok=True)
    download_dir = run_artifacts_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    window_dir = run_artifacts_dir / "rolling_windows" if mode == "rolling_extract" else None

    for result in results:
        blob_path = _persist_blob_for_result(result, download_dir)
        metadata = dict(result.metadata or {})
        metadata["download_mode"] = mode
        result.metadata = metadata

        if mode == "download_only" and blob_path is not None:
            result.text = ""
        elif mode == "rolling_extract" and window_dir is not None:
            _attach_rolling_windows(result, window_dir, window_size, window_step, max_windows)

        result.raw_bytes = None


def _read_raw_text_for_extraction(result: FetchResult) -> str:
    """Return the best available raw payload for extraction.

    Preference order:
      - inline `result.text`
      - `result.raw_bytes` decoded as utf-8
      - `metadata[text_path]` or `metadata[file_path]` if it exists on disk
      - `metadata[blob_path]` if it exists on disk
    """

    if result.text:
        return result.text
    if result.raw_bytes:
        try:
            return result.raw_bytes.decode("utf-8", "ignore")
        except Exception:
            pass
    metadata = result.metadata or {}
    for key in (K_TEXT_PATH, "file_path", "blob_path"):
        raw = str(metadata.get(key) or "").strip()
        if not raw:
            continue
        path = Path(raw)
        if not path.is_absolute():
            # Best-effort: treat relative paths as cwd-relative
            path = Path.cwd() / path
        try:
            if path.exists():
                return path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
    return ""


def materialize_extracted_text(
    results: Iterable[FetchResult],
    run_artifacts_dir: Path,
    *,
    enabled: bool = True,
    min_chars: int = 1,
) -> None:
    """Persist extracted/clean text as a first-class artifact.

    This makes it trivial for humans/agents to inspect whether a fetched URL is
    usable without re-running extraction logic elsewhere.
    """

    if not enabled:
        return
    out_dir = run_artifacts_dir / "extracted_text"
    out_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        metadata = dict(result.metadata or {})
        if (metadata.get("content_verdict") or "").lower() != "ok":
            continue
        raw = _read_raw_text_for_extraction(result)
        if not raw.strip():
            continue
        assessment = extract_content_features(raw, result.content_type or "text/html", result.url)
        extracted = (assessment.text or "").strip()
        if len(extracted) < max(1, int(min_chars)):
            continue
        payload = extracted.encode("utf-8")
        sha = hashlib.sha256(payload).hexdigest()
        path = out_dir / f"{sha}.txt"
        if not path.exists():
            path.write_text(extracted + "\n", encoding="utf-8")
        metadata["extracted_text_path"] = str(path)
        metadata["extracted_text_sha256"] = sha
        metadata["extracted_text_len"] = len(extracted)
        metadata["extracted_text_source"] = assessment.source
        result.metadata = metadata


def materialize_markdown(
    results: Iterable[FetchResult],
    run_artifacts_dir: Path,
    *,
    enabled: bool = False,
    min_chars: int = 1,
    emit_fit_markdown: bool = True,
    fit_min_chars: int = 200,
    overrides_path: Optional[Path] = None,
) -> None:
    """Persist an LLM-friendly markdown artifact for HTML-ish pages.

    This is opt-in because it is more expensive than writing raw blobs.
    """

    if not enabled:
        return
    if trafilatura is None:
        return
    out_dir = run_artifacts_dir / "markdown"
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_dir = run_artifacts_dir / "fit_markdown"
    fit_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        metadata = dict(result.metadata or {})
        if (metadata.get("content_verdict") or "").lower() != "ok":
            continue
        raw = _read_raw_text_for_extraction(result)
        if not raw.strip():
            continue
        # Only attempt markdown conversion when the payload is likely HTML.
        ct = (result.content_type or "").lower()
        looks_html = ("html" in ct) or ("<html" in raw.lower()) or ("</" in raw and "<" in raw)
        if not looks_html:
            continue
        try:
            md = trafilatura.extract(
                raw,
                url=result.url,
                include_tables=True,
                favor_recall=True,
                output_format="markdown",
            )
        except Exception:
            md = None
        md = (md or "").strip()
        if len(md) < max(1, int(min_chars)):
            continue
        payload = md.encode("utf-8")
        sha = hashlib.sha256(payload).hexdigest()
        path = out_dir / f"{sha}.md"
        if not path.exists():
            path.write_text(md + "\n", encoding="utf-8")
        metadata["markdown_path"] = str(path)
        metadata["markdown_sha256"] = sha
        metadata["markdown_len"] = len(md)
        metadata["markdown_backend"] = "trafilatura"
        result.metadata = metadata

        if emit_fit_markdown:
            selectors = _pick_fit_markdown_selectors(result.url, overrides_path)
            reduced_html = _select_html_by_css(raw, selectors) if selectors else raw
            try:
                fit_md = trafilatura.extract(
                    reduced_html,
                    url=result.url,
                    include_tables=True,
                    favor_recall=True,
                    output_format="markdown",
                )
            except Exception:
                fit_md = None
            fit_md = _prune_markdown_to_first_h1((fit_md or "").strip())
            if len(fit_md) >= max(1, int(fit_min_chars)):
                fit_payload = fit_md.encode("utf-8")
                fit_sha = hashlib.sha256(fit_payload).hexdigest()
                fit_path = fit_dir / f"{fit_sha}.md"
                if not fit_path.exists():
                    fit_path.write_text(fit_md + "\n", encoding="utf-8")
                metadata = dict(result.metadata or {})
                metadata["fit_markdown_path"] = str(fit_path)
                metadata["fit_markdown_sha256"] = fit_sha
                metadata["fit_markdown_len"] = len(fit_md)
                metadata["fit_markdown_backend"] = "trafilatura_prune"
                if selectors:
                    metadata["fit_markdown_selectors"] = selectors
                # "usable" heuristic: long enough and contains a top-level heading.
                metadata["fit_markdown_usable"] = bool(fit_md.startswith("# ") and len(fit_md) >= max(1, int(fit_min_chars)))
                result.metadata = metadata


def annotate_paywall_metadata(results: Iterable[FetchResult], policy) -> None:
    for result in results:
        domain = result.domain or urlparse(result.url).netloc
        if domain and policy:
            safe_suffixes = getattr(policy, "paywall_safe_suffixes", tuple())
            strip = set(getattr(policy, "strip_subdomains", frozenset({"www"})))
            if _is_safe_domain(domain, policy.paywall_safe_domains, safe_suffixes, strip):
                metadata = dict(result.metadata or {})
                metadata["paywall_detection"] = {"verdict": "safe", "score": 0.0}
                metadata["paywall_verdict"] = "safe"
                metadata["paywall_score"] = 0.0
                result.metadata = metadata
                continue
        # If body is empty, still mark paywall suspicion when domain/status hints match.
        if not result.text:
            metadata = dict(result.metadata or {})
            suspected = False
            if domain and policy and domain in getattr(policy, "paywall_domains", set()):
                suspected = True
            if getattr(result, "status", None) in getattr(policy, "paywall_status_codes", set()):
                suspected = True
            if suspected:
                detection = {
                    "verdict": "maybe",
                    "score": 0.6,
                    "indicators": {"no_text_body": True},
                }
                metadata["paywall_detection"] = detection
                metadata["paywall_verdict"] = detection["verdict"]
                metadata["paywall_score"] = detection["score"]
                result.metadata = metadata
            continue
        ct = (result.content_type or "").lower()
        if "html" not in ct:
            continue
        try:
            detection = detect_paywall(
                url=result.url,
                status=result.status,
                html=result.text,
                policy=policy,
            )
        except Exception:
            continue
        metadata = dict(result.metadata or {})
        metadata["paywall_detection"] = detection
        metadata["paywall_verdict"] = detection.get("verdict")
        metadata["paywall_score"] = detection.get("score")
        result.metadata = metadata


def sanity_check() -> None:
    sample = "Fetcher sanity sentence one. Sentence two ensures overlap."
    windows = build_rolling_windows(sample, window_size=64, window_step=32, max_windows=0)
    assert windows, "Rolling window builder returned no windows for sample text"
    assert windows[0]["end"] > windows[0]["start"]


sanity_check()

__all__ = [
    "annotate_paywall_metadata",
    "apply_download_mode",
    "build_rolling_windows",
    "materialize_extracted_text",
    "materialize_markdown",
    "maybe_externalize_text",
    "sanity_check",
]
