import re

import pytest

from fetcher.consumer import generate_run_id, parse_manifest_lines, _render_walkthrough


def test_parse_manifest_lines_allows_comments_and_blank():
    lines = [
        "# comment",
        "",
        "https://example.com/a",
        "https://example.com/b",
        "   ",
    ]
    assert parse_manifest_lines(lines) == ["https://example.com/a", "https://example.com/b"]


def test_parse_manifest_lines_rejects_inline_metadata():
    lines = ["https://example.com/a # nope"]
    with pytest.raises(ValueError):
        parse_manifest_lines(lines)


def test_generate_run_id_format():
    run_id = generate_run_id()
    assert re.match(r"^\d{8}T\d{6}Z_[0-9a-f]{6}$", run_id)


def test_render_walkthrough_deterministic():
    summary = {
        "run_id": "20260102T153045Z_a1b2c3",
        "started_at": "2026-01-02T15:30:45Z",
        "finished_at": "2026-01-02T15:31:12Z",
        "duration_ms": 27000,
        "soft_failures": ["brave_api_key_missing"],
        "counts": {
            "total": 1,
            "downloaded": 1,
            "ok": 1,
            "failed": 0,
            "used_playwright": 0,
            "used_alternates": 0,
        },
        "items": [
            {
                "original_url": "https://example.com/a",
                "requested_url": "https://example.com/a",
                "final_downloaded_url": "https://example.com/a",
                "status": 200,
                "verdict": "ok",
                "paywall_verdict": "ok",
                "artifacts": {
                    "download_path": "run/artifacts/x/downloads/abc.html",
                    "extracted_text_path": "run/artifacts/x/extracted_text/abc.txt",
                    "markdown_path": "run/artifacts/x/markdown/abc.md",
                    "fit_markdown_path": None,
                },
                "warnings": [],
                "errors": [],
            }
        ],
    }
    output = _render_walkthrough(summary)
    assert output.startswith("# Walkthrough")
    assert "Run ID: 20260102T153045Z_a1b2c3" in output
    assert "Soft Failures" in output
    assert "## Item 1" in output
