from urllib.parse import urlparse

import pytest

try:  # Optional during minimal installs
    import fitz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fitz = None  # type: ignore

from fetcher.workflows.extract_utils import evaluate_result_content
from fetcher.workflows.web_fetch import FetchConfig, FetchResult, URLFetcher


def _result(html: str, content_type: str = "text/html", url: str = "https://example.com") -> FetchResult:
    return FetchResult(
        url=url,
        domain=urlparse(url).hostname or "example.com",
        status=200,
        content_type=content_type,
        text=html,
        fetched_at="now",
        method="aiohttp",
        metadata={},
        from_cache=False,
        raw_bytes=None,
    )


def test_evaluate_result_marks_thin_content_as_failure():
    html = "<html><body><p>Subscribe to continue</p><button>Log in</button></body></html>"
    result = _result(html)
    evaluate_result_content(result)
    assert result.metadata["content_verdict"] in {"paywall", "thin"}
    if result.metadata["content_verdict"] == "paywall":
        assert result.metadata.get("paywall_stub") is True
        assert result.text == html
    else:
        assert result.text == ""


def test_evaluate_result_accepts_article_content():
    body = "<p>" + ("This is a sentence. " * 200) + "</p>"
    html = f"<html><body><article>{body}</article></body></html>"
    result = _result(html)
    evaluate_result_content(result)
    assert result.metadata["content_verdict"] == "ok"
    # result.text should now contain extracted text, not raw HTML
    assert "This is a sentence" in result.text
    assert "<html>" not in result.text


def test_evaluate_result_allows_link_heavy_domains():
    links = "".join([f'<a href="/mitigations/M10{i}">M10{i}</a>' for i in range(50)])
    html = f"<html><body><div>{links}</div></body></html>"
    result = _result(html, url="https://attack.mitre.org/mitigations/M1032/")
    evaluate_result_content(result)
    assert result.metadata["content_verdict"] == "ok"


def test_evaluate_result_marks_deprecation_warning():
    warning = """
    <div class=\"card danger-card\">
        <div class=\"card-header\"><h5>Deprecation Warning</h5></div>
        <div class=\"card-body\"><p>This data source is deprecated.</p></div>
    </div>
    """
    body = "<p>" + ("Valid text. " * 200) + "</p>"
    html = f"<html><body>{warning}{body}</body></html>"
    result = _result(html)
    evaluate_result_content(result)
    assert result.metadata["content_verdict"] == "ok"
    assert result.metadata["deprecation_warning"] is True
    assert "Deprecation Warning" in result.metadata["deprecation_warning_text"]


def test_evaluate_result_handles_structured_definitions():
    html = (
        "<html><body>"
        "<div class='glossary'>"
        "<strong>Definitions:</strong>"
        "<div class='indent-1'><p>Techniques to counter traffic analysis.</p></div>"
        "</div>"
        "</body></html>"
    )
    result = _result(html, url="https://csrc.nist.gov/glossary/term/traffic_flow_security")
    evaluate_result_content(result)
    assert result.metadata["content_verdict"] == "ok"
    assert result.metadata.get("structured_definition") is True


def test_capec_registration_pages_marked_paywall():
    html = "<html><body><form><input name='email'></form></body></html>"
    result = _result(html, url="https://capec.mitre.org/community/registration.html")
    evaluate_result_content(result)
    assert result.metadata["content_verdict"] == "paywall"
    assert "registration_required" in result.metadata["content_reasons"]


def test_capec_community_portal_reason_added():
    html = "<html><body><h1>CAPEC Community</h1><p>Announcements.</p></body></html>"
    result = _result(html, url="https://capec.mitre.org/community/index.html")
    evaluate_result_content(result)
    assert "community_portal" in result.metadata["content_reasons"]


def test_pdf_content_not_marked_paywall_for_long_reports():
    # Simulate a long-form PDF body that happens to contain subscription-ish
    # language; ensure it is treated as usable content, not paywall/junk.
    body = "This report is public. " * 120
    # Include a couple of generic markers that would trip HTML heuristics.
    body += " Please subscribe to our newsletter and log in to your account for updates."
    result = _result(body, content_type="application/pdf", url="https://example.com/report.pdf")
    # PDF extraction path sets this flag in real runs; mirror it here.
    result.metadata = {"pdf_text_extracted": True, "pdf_pages": 50, "pdf_characters": len(body)}

    evaluate_result_content(result)

    assert result.metadata["content_verdict"] == "ok"
    # PDF text should be preserved so downstream chunking can run.
    assert result.text == body


def test_pdf_password_protected_sets_verdict_and_clears_text():
    result = _result("", content_type="application/pdf", url="https://example.com/protected.pdf")
    result.metadata = {"pdf_password_protected": True}

    evaluate_result_content(result)

    assert result.metadata["content_verdict"] == "password_protected"
    assert result.metadata["content_text_len"] == 0
    assert "pdf_password_protected" in result.metadata["content_reasons"]
    assert result.text == ""


def test_pdf_password_recovered_allows_content():
    body = "Recovered text " * 120
    result = _result(body, content_type="application/pdf", url="https://example.com/recovered.pdf")
    result.metadata = {"pdf_password_protected": True, "pdf_password_recovered": True}

    evaluate_result_content(result)

    assert result.metadata["content_verdict"] == "ok"
    # For non-HTML content types, text is stripped but otherwise preserved
    assert result.text == body.strip()


def test_extract_pdf_text_flags_password_protected_pdf_bytes():
    """Verify password-protected PDFs raise RuntimeError when password cannot be recovered."""
    if fitz is None:
        pytest.skip("PyMuPDF not installed")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Secret text")
    permissions = getattr(fitz, "PDF_PERM_NONE", 0)
    pdf_bytes = doc.tobytes(
        encryption=getattr(fitz, "PDF_ENCRYPT_AES_256", 4),
        owner_pw="owner",
        user_pw="owner",
        permissions=permissions,
    )
    doc.close()

    fetcher = URLFetcher(FetchConfig())
    # Password-protected PDFs that can't be cracked now raise RuntimeError
    # (changed from silent failure to loud failure for observability)
    with pytest.raises(RuntimeError) as exc_info:
        fetcher._extract_pdf_text(pdf_bytes, "https://example.com/protected.pdf")

    assert "PDF password not recovered" in str(exc_info.value)
    assert "protected.pdf" in str(exc_info.value)


def test_extract_pdf_text_cracks_simple_password(monkeypatch):
    if fitz is None:
        pytest.skip("PyMuPDF not installed")
    pytest.importorskip("pdferli")

    monkeypatch.setenv("FETCHER_PDF_CRACK_ENABLE", "1")
    monkeypatch.setenv("FETCHER_PDF_CRACK_CHARSET", "0123456789")
    monkeypatch.setenv("FETCHER_PDF_CRACK_MINLEN", "4")
    monkeypatch.setenv("FETCHER_PDF_CRACK_MAXLEN", "4")
    monkeypatch.setenv("FETCHER_PDF_CRACK_TIMEOUT", "10")
    monkeypatch.setenv("FETCHER_PDF_CRACK_PROCESSES", "2")

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Unlocked content")
    pdf_bytes = doc.tobytes(
        encryption=getattr(fitz, "PDF_ENCRYPT_AES_256", 4),
        owner_pw="1234",
        user_pw="1234",
        permissions=getattr(fitz, "PDF_PERM_NONE", 0),
    )
    doc.close()

    fetcher = URLFetcher(FetchConfig())
    text, meta = fetcher._extract_pdf_text(pdf_bytes, "https://example.com/locked.pdf")

    assert "Unlocked content" in text
    assert meta.get("pdf_password_recovered") is True
    assert meta.get("pdf_password_protected") is True
    assert meta.get("pdf_text_extracted") is True


def test_pdf_soft_404_detected_and_marked_missing():
    text = "Sorry, the page you were looking for can't be found."
    result = _result(text, content_type="application/pdf", url="https://cm.scholasticahq.com/article/5906.pdf")
    result.metadata = {"pdf_text_extracted": True}

    evaluate_result_content(result)

    assert result.metadata["content_verdict"] == "missing_file"
    assert "pdf_soft_404" in result.metadata["content_reasons"]
    assert result.text == ""


def test_pdf_url_returning_html_marked_missing():
    html = "<html><body><h1>Sorry, the page you were looking for can't be found.</h1></body></html>"
    result = _result(html, content_type="text/html", url="https://example.com/missing.pdf")
    # No pdf_text_extracted marker; mimic an HTML placeholder response for a .pdf URL
    evaluate_result_content(result)

    assert result.metadata["content_verdict"] == "missing_file"
    assert "pdf_fetch_mismatch" in result.metadata["content_reasons"] or "pdf_soft_404" in result.metadata["content_reasons"]
    assert result.text == ""


def test_pdf_access_denied_edgesuite_marked_missing():
    html = "<html><body><h1>Access Denied</h1><p>See https://errors.edgesuite.net/123</p></body></html>"
    result = _result(html, content_type="text/html", url="https://www.nsa.gov/foo.pdf")

    evaluate_result_content(result)

    assert result.metadata["content_verdict"] == "missing_file"
    assert any("pdf_soft_404" in r or "pdf_fetch_mismatch" in r for r in result.metadata["content_reasons"])
    assert result.text == ""
