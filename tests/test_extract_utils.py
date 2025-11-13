from fetcher.workflows.extract_utils import evaluate_result_content
from fetcher.workflows.web_fetch import FetchResult


def _result(html: str, content_type: str = "text/html") -> FetchResult:
    return FetchResult(
        url="https://example.com",
        domain="example.com",
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
    assert result.text == ""


def test_evaluate_result_accepts_article_content():
    body = "<p>" + ("This is a sentence. " * 200) + "</p>"
    html = f"<html><body><article>{body}</article></body></html>"
    result = _result(html)
    evaluate_result_content(result)
    assert result.metadata["content_verdict"] == "ok"
    assert result.text == html
