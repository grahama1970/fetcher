#!/usr/bin/env python3
"""Compare fetched HTML to emitted markdown (plus Playwright screenshot).

This is an integration-style diagnostic tool intended for humans/project-agents.
It fetches a URL via fetcher, persists:
  - raw HTML blob (`blob_path`)
  - extracted text (`extracted_text_path`, if enabled)
  - markdown (`markdown_path`, enabled for this run)
  - fit markdown (`fit_markdown_path`, pruned variant when enabled)
  - Playwright screenshot (`playwright_screenshot`, when Playwright is used)

Then it writes a small report under the run artifacts directory:
  - comparison.json (machine-readable)
  - comparison.md (human-readable)

Usage:
  uv run python scripts/compare_markdown_to_html.py --url https://d3fend.mitre.org/technique/d3f:FileEviction/
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Set

from bs4 import BeautifulSoup


@dataclass(frozen=True, slots=True)
class HeadingReport:
    html_h1: List[str]
    html_h2: List[str]
    html_h3: List[str]
    md_h1: List[str]
    md_h2: List[str]
    md_h3: List[str]
    overlap_h1: List[str]
    overlap_h2: List[str]
    overlap_h3: List[str]
    fit_md_h1: List[str]
    fit_md_h2: List[str]
    fit_md_h3: List[str]
    fit_overlap_h1: List[str]
    fit_overlap_h2: List[str]
    fit_overlap_h3: List[str]


def _extract_html_headings(html: str) -> tuple[list[str], list[str], list[str]]:
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return [], [], []
    h1 = [t.get_text(" ", strip=True) for t in soup.find_all("h1") if t.get_text(" ", strip=True)]
    h2 = [t.get_text(" ", strip=True) for t in soup.find_all("h2") if t.get_text(" ", strip=True)]
    h3 = [t.get_text(" ", strip=True) for t in soup.find_all("h3") if t.get_text(" ", strip=True)]
    return h1[:25], h2[:50], h3[:75]


def _extract_md_headings(md: str) -> tuple[list[str], list[str], list[str]]:
    h1: list[str] = []
    h2: list[str] = []
    h3: list[str] = []
    for line in (md or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            h1.append(line[2:].strip())
        elif line.startswith("## "):
            h2.append(line[3:].strip())
        elif line.startswith("### "):
            h3.append(line[4:].strip())
    return h1[:25], h2[:50], h3[:75]


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"[A-Za-z0-9]{3,}", text or "")
    return {t.lower() for t in tokens}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return round(inter / max(1, union), 4)


def _write_report(run_dir: Path, payload: dict) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "comparison.json"
    md_path = run_dir / "comparison.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Human-friendly summary
    lines = []
    lines.append("# Fetcher HTML vs Markdown comparison\n\n")
    lines.append(f"- url: `{payload.get('url')}`\n")
    lines.append(f"- status: `{payload.get('status')}`\n")
    lines.append(f"- method: `{payload.get('method')}`\n")
    lines.append(f"- usable: `{payload.get('usable')}` junk: `{payload.get('is_junk')}` reason: `{payload.get('junk_reason')}`\n")
    for key in ("blob_path", "markdown_path", "fit_markdown_path", "extracted_text_path", "playwright_screenshot"):
        if payload.get(key):
            lines.append(f"- {key}: `{payload.get(key)}`\n")
    lines.append("\n## Quick checks\n\n")
    lines.append(f"- comparison_ok: `{payload.get('comparison_ok')}`\n")
    lines.append(f"- html_chars: `{payload.get('html_chars')}`\n")
    lines.append(f"- markdown_chars: `{payload.get('markdown_chars')}`\n")
    lines.append(f"- markdown_usable: `{payload.get('markdown_usable')}`\n")
    lines.append(f"- fit_markdown_chars: `{payload.get('fit_markdown_chars')}`\n")
    lines.append(f"- fit_markdown_usable: `{payload.get('fit_markdown_usable')}`\n")
    lines.append(f"- html_vs_markdown_token_jaccard: `{payload.get('html_vs_markdown_token_jaccard')}`\n")
    lines.append(f"- html_vs_fit_markdown_token_jaccard: `{payload.get('html_vs_fit_markdown_token_jaccard')}`\n")
    lines.append(f"- fit_equals_markdown: `{payload.get('fit_equals_markdown')}`\n")
    if payload.get("fit_markdown_selectors"):
        lines.append(f"- fit_markdown_selectors: `{payload.get('fit_markdown_selectors')}`\n")
    lines.append("\n## Headings overlap\n\n")
    headings = payload.get("headings") or {}
    lines.append(f"- h1 overlap: `{len(headings.get('overlap_h1', []))}`\n")
    lines.append(f"- h2 overlap: `{len(headings.get('overlap_h2', []))}`\n")
    lines.append(f"- h3 overlap: `{len(headings.get('overlap_h3', []))}`\n")
    lines.append("\n### HTML h1\n\n")
    for item in headings.get("html_h1", [])[:10]:
        lines.append(f"- {item}\n")
    lines.append("\n### Markdown h1\n\n")
    for item in headings.get("md_h1", [])[:10]:
        lines.append(f"- {item}\n")
    lines.append("\n### Fit Markdown h1\n\n")
    for item in headings.get("fit_md_h1", [])[:10]:
        lines.append(f"- {item}\n")
    md_path.write_text("".join(lines), encoding="utf-8")
    return json_path, md_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--run-dir", default="run/artifacts/compare_markdown")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--markdown-min-chars", type=int, default=int(os.getenv("FETCHER_COMPARE_MARKDOWN_MIN_CHARS", "400")))
    parser.add_argument("--fit-markdown-min-chars", type=int, default=int(os.getenv("FETCHER_COMPARE_FIT_MARKDOWN_MIN_CHARS", "400")))
    args = parser.parse_args(argv)

    # Force markdown emission for this run (policy is read at import time).
    os.environ["FETCHER_EMIT_MARKDOWN"] = "1"
    os.environ.setdefault("FETCHER_EMIT_EXTRACTED_TEXT", "1")

    from fetcher.workflows.fetcher import fetch_url
    from fetcher.workflows.web_fetch import FetchConfig

    run_dir = Path(args.run_dir)
    cfg = FetchConfig(concurrency=1, per_domain=1, timeout=float(args.timeout))
    # Keep single-url fetches bounded.
    cfg.enable_toc_fanout = False

    result = fetch_url(
        args.url,
        fetch_config=cfg,
        run_artifacts_dir=run_dir,
        download_mode="download_only",
    )
    meta = result.metadata or {}
    blob_path = meta.get("blob_path")
    markdown_path = meta.get("markdown_path")
    fit_markdown_path = meta.get("fit_markdown_path")
    screenshot_path = meta.get("playwright_screenshot")
    extracted_text_path = meta.get("extracted_text_path")

    html = ""
    if blob_path and Path(blob_path).exists():
        html = Path(blob_path).read_text(encoding="utf-8", errors="ignore")
    md = ""
    if markdown_path and Path(markdown_path).exists():
        md = Path(markdown_path).read_text(encoding="utf-8", errors="ignore")
    fit_md = ""
    if fit_markdown_path and Path(fit_markdown_path).exists():
        fit_md = Path(fit_markdown_path).read_text(encoding="utf-8", errors="ignore")

    html_h1, html_h2, html_h3 = _extract_html_headings(html)
    md_h1, md_h2, md_h3 = _extract_md_headings(md)
    fit_h1, fit_h2, fit_h3 = _extract_md_headings(fit_md)

    overlap_h1 = sorted(set(html_h1) & set(md_h1))
    overlap_h2 = sorted(set(html_h2) & set(md_h2))
    overlap_h3 = sorted(set(html_h3) & set(md_h3))
    fit_overlap_h1 = sorted(set(html_h1) & set(fit_h1))
    fit_overlap_h2 = sorted(set(html_h2) & set(fit_h2))
    fit_overlap_h3 = sorted(set(html_h3) & set(fit_h3))
    headings = HeadingReport(
        html_h1=html_h1,
        html_h2=html_h2,
        html_h3=html_h3,
        md_h1=md_h1,
        md_h2=md_h2,
        md_h3=md_h3,
        overlap_h1=overlap_h1,
        overlap_h2=overlap_h2,
        overlap_h3=overlap_h3,
        fit_md_h1=fit_h1,
        fit_md_h2=fit_h2,
        fit_md_h3=fit_h3,
        fit_overlap_h1=fit_overlap_h1,
        fit_overlap_h2=fit_overlap_h2,
        fit_overlap_h3=fit_overlap_h3,
    )

    html_usable = bool(meta.get("usable"))
    markdown_usable = bool(markdown_path and md.strip() and (len(md) >= max(1, int(args.markdown_min_chars))) and md_h1)
    fit_markdown_usable = bool(
        fit_markdown_path and fit_md.strip() and (len(fit_md) >= max(1, int(args.fit_markdown_min_chars))) and fit_h1
    )
    # When Playwright is used, screenshot presence is part of the "debuggable" contract.
    screenshot_ok = True
    if str(result.method or "").lower() == "playwright":
        screenshot_ok = bool(screenshot_path and Path(screenshot_path).exists())

    payload = {
        "url": result.url,
        "status": result.status,
        "method": result.method,
        "usable": meta.get("usable"),
        "is_junk": meta.get("is_junk"),
        "junk_reason": meta.get("junk_reason"),
        "blob_path": blob_path,
        "markdown_path": markdown_path,
        "fit_markdown_path": fit_markdown_path,
        "extracted_text_path": extracted_text_path,
        "playwright_screenshot": screenshot_path,
        "html_chars": len(html),
        "markdown_chars": len(md),
        "markdown_usable": markdown_usable,
        "fit_markdown_chars": len(fit_md),
        "fit_markdown_usable": fit_markdown_usable,
        "fit_markdown_selectors": meta.get("fit_markdown_selectors"),
        "html_vs_markdown_token_jaccard": _jaccard(_tokenize(html), _tokenize(md)),
        "html_vs_fit_markdown_token_jaccard": _jaccard(_tokenize(html), _tokenize(fit_md)),
        "headings": asdict(headings),
    }
    payload["fit_equals_markdown"] = bool(md.strip() and fit_md.strip() and (md.strip() == fit_md.strip()))
    payload["comparison_ok"] = bool(html_usable and (markdown_usable or fit_markdown_usable) and screenshot_ok)
    json_path, md_path = _write_report(run_dir, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
