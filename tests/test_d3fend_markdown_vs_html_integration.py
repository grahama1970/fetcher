import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.getenv("FETCHER_INTEGRATION_TESTS", "0") != "1", reason="integration test (network + playwright)")
def test_d3fend_markdown_matches_html_artifacts(tmp_path: Path) -> None:
    os.environ["FETCHER_EMIT_MARKDOWN"] = "1"
    os.environ["FETCHER_EMIT_EXTRACTED_TEXT"] = "1"

    from fetcher.workflows.fetcher import fetch_url
    from fetcher.workflows.web_fetch import FetchConfig

    url = "https://d3fend.mitre.org/technique/d3f:FileEviction/"
    cfg = FetchConfig(concurrency=1, per_domain=1, timeout=30)
    cfg.enable_toc_fanout = False

    result = fetch_url(url, fetch_config=cfg, run_artifacts_dir=tmp_path, download_mode="download_only")
    meta = result.metadata or {}

    assert result.status == 200
    assert meta.get("usable") is True
    assert meta.get("is_junk") is False

    blob_path = meta.get("blob_path")
    assert blob_path, "expected blob_path when download_only"
    assert Path(blob_path).exists()

    markdown_path = meta.get("markdown_path")
    assert markdown_path, "expected markdown_path when FETCHER_EMIT_MARKDOWN=1"
    assert Path(markdown_path).exists()

    fit_markdown_path = meta.get("fit_markdown_path")
    assert fit_markdown_path, "expected fit_markdown_path when FETCHER_EMIT_MARKDOWN=1 and selector overrides exist"
    assert Path(fit_markdown_path).exists()

    extracted_text_path = meta.get("extracted_text_path")
    assert extracted_text_path, "expected extracted_text_path when FETCHER_EMIT_EXTRACTED_TEXT=1"
    assert Path(extracted_text_path).exists()

    # Screenshot presence confirms Playwright actually ran.
    screenshot = meta.get("playwright_screenshot")
    assert screenshot, "expected playwright_screenshot for D3FEND (SPA) pages"
    assert Path(screenshot).exists()

    md = Path(markdown_path).read_text(encoding="utf-8", errors="ignore")
    assert "# File Eviction" in md
    assert "## Definition" in md
    assert "## How it works" in md
