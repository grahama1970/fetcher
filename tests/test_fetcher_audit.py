import json
from fetcher.workflows import fetcher
from fetcher.workflows.web_fetch import FetchResult
from fetcher.workflows.download_utils import materialize_extracted_text, materialize_markdown
from fetcher.workflows.extract_utils import evaluate_result_content
from pathlib import Path


def test_collect_pdf_paywall_mismatches_flags_extracted_pdf() -> None:
    text = "Public report content."
    result = FetchResult(
        url="https://example.com/report.pdf",
        domain="example.com",
        status=200,
        content_type="application/pdf",
        text=text,
        fetched_at="now",
        method="pdf",
        metadata={
            "pdf_text_extracted": True,
            "pdf_pages": 10,
            "pdf_characters": len(text),
            "paywall_verdict": "likely",
            "paywall_score": 1.0,
            "content_verdict": "ok",
        },
        from_cache=False,
        raw_bytes=None,
    )

    mismatches = fetcher._collect_pdf_paywall_mismatches([result])  # type: ignore[attr-defined]
    assert mismatches
    assert mismatches[0]["url"] == result.url
    assert mismatches[0]["pdf_pages"] == 10


def test_collect_pdf_paywall_mismatches_ignores_non_paywall_pdfs() -> None:
    text = "Another public report."
    result = FetchResult(
        url="https://example.com/ok.pdf",
        domain="example.com",
        status=200,
        content_type="application/pdf",
        text=text,
        fetched_at="now",
        method="pdf",
        metadata={
            "pdf_text_extracted": True,
            "pdf_pages": 5,
            "pdf_characters": len(text),
            "paywall_verdict": "unlikely",
            "content_verdict": "ok",
        },
        from_cache=False,
        raw_bytes=None,
    )

    mismatches = fetcher._collect_pdf_paywall_mismatches([result])  # type: ignore[attr-defined]
    assert mismatches == []


def test_materialize_extracted_text_writes_artifact(tmp_path: Path) -> None:
    big = "Hello world. " * 200
    html = f"<html><body><main><h1>Title</h1><p>{big}</p><p>{big}</p></main></body></html>"
    result = FetchResult(
        url="https://example.com/page",
        domain="example.com",
        status=200,
        content_type="text/html",
        text=html,
        fetched_at="now",
        method="aiohttp",
        metadata={},
        from_cache=False,
        raw_bytes=None,
    )
    evaluate_result_content(result)
    materialize_extracted_text([result], tmp_path, enabled=True, min_chars=1)
    md = result.metadata or {}
    assert md.get("extracted_text_path")
    p = Path(md["extracted_text_path"])
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "Hello world" in body


def test_materialize_markdown_writes_artifact(tmp_path: Path) -> None:
    big = "Hello world. " * 200
    html = f"<html><body><main><h1>Title</h1><p>{big}</p><p>{big}</p></main></body></html>"
    result = FetchResult(
        url="https://example.com/page",
        domain="example.com",
        status=200,
        content_type="text/html",
        text=html,
        fetched_at="now",
        method="aiohttp",
        metadata={},
        from_cache=False,
        raw_bytes=html.encode("utf-8"),  # materialize_markdown needs raw HTML
    )
    evaluate_result_content(result)
    materialize_markdown([result], tmp_path, enabled=True, min_chars=1, emit_fit_markdown=True, fit_min_chars=50, overrides_path=None)
    md = result.metadata or {}
    assert md.get("markdown_path")
    p = Path(md["markdown_path"])
    assert p.exists()
    body = p.read_text(encoding="utf-8")
    assert "Hello world" in body


def test_fit_markdown_uses_selector_rules(tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            [
                {
                    "domain": "example.com",
                    "path_prefix": "/page",
                    "fit_markdown_selectors": ["main"],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    big = "Body text. " * 300
    html = f"<html><body><header>BOILERPLATE NAV</header><main><h1>Title</h1><p>{big}</p></main></body></html>"
    result = FetchResult(
        url="https://example.com/page",
        domain="example.com",
        status=200,
        content_type="text/html",
        text=html,
        fetched_at="now",
        method="aiohttp",
        metadata={},
        from_cache=False,
        raw_bytes=html.encode("utf-8"),  # materialize_markdown needs raw HTML
    )
    evaluate_result_content(result)
    materialize_markdown([result], tmp_path, enabled=True, min_chars=1, emit_fit_markdown=True, fit_min_chars=1, overrides_path=overrides)
    md = result.metadata or {}
    assert md.get("fit_markdown_path")
    fit = Path(md["fit_markdown_path"]).read_text(encoding="utf-8", errors="ignore")
    assert "BOILERPLATE NAV" not in fit
    assert "Body text" in fit


def test_write_junk_report_emits_table(tmp_path: Path) -> None:
    bad = FetchResult(
        url="https://example.com/bad",
        domain="example.com",
        status=200,
        content_type="text/html",
        text="<html><body>hi</body></html>",
        fetched_at="now",
        method="aiohttp",
        metadata={},
        from_cache=False,
        raw_bytes=None,
    )
    evaluate_result_content(bad)
    fetcher._write_junk_report([bad], tmp_path)  # type: ignore[attr-defined]
    assert (tmp_path / "junk_results.jsonl").exists()
    assert (tmp_path / "junk_summary.json").exists()
    assert (tmp_path / "junk_table.md").exists()
