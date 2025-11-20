"""Helper utilities for externalizing bodies, rolling windows, and paywall metadata."""

from __future__ import annotations

import hashlib
import json
import os
import re
import mimetypes
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

try:  # spaCy is optional; fall back to regex segmentation when missing
    import spacy  # type: ignore
except Exception:  # pragma: no cover - spaCy optional
    spacy = None  # type: ignore

from .paywall_detector import detect_paywall
from .extract_utils import verify_blob_content
from .fetcher_utils import is_safe_domain as _is_safe_domain
from .web_fetch import FetchResult
from ..core.keys import K_TEXT_PATH

_SPACY_SENTENCIZER = None


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

    for result in results:
        text = result.text or ""
        if not text:
            continue
        text_bytes = text.encode("utf-8")
        if max_inline_bytes > 0 and len(text_bytes) <= max_inline_bytes:
            continue

        sha = hashlib.sha256(text_bytes).hexdigest()
        blob_path = text_root / f"{sha}.txt"
        blob_path.write_text(text, encoding="utf-8")

        metadata = dict(result.metadata or {})
        metadata[K_TEXT_PATH] = str(blob_path)
        metadata["text_externalized"] = True
        metadata["text_inline_bytes"] = len(text_bytes)
        metadata["text_sha256"] = sha

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
        if not result.text:
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
    "maybe_externalize_text",
    "sanity_check",
]
