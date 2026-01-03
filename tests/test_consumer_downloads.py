from datetime import datetime, timezone

from fetcher.workflows.download_utils import persist_downloads
from fetcher.workflows.web_fetch import FetchResult


def _make_result(url: str, *, text: str = "", raw_bytes: bytes | None = None, verdict: str = "paywall") -> FetchResult:
    return FetchResult(
        url=url,
        domain="example.com",
        status=200,
        content_type="text/html",
        text=text,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        method="aiohttp",
        metadata={"content_verdict": verdict},
        raw_bytes=raw_bytes,
    )


def test_persist_downloads_uses_raw_bytes(tmp_path):
    result = _make_result("https://example.com/raw", raw_bytes=b"payload")
    errors = persist_downloads([result], tmp_path, allow_junk=True)
    assert errors["https://example.com/raw"] is None
    blob_path = result.metadata.get("blob_path")
    assert blob_path
    with open(blob_path, "rb") as fh:
        assert fh.read() == b"payload"


def test_persist_downloads_uses_text_when_no_bytes(tmp_path):
    result = _make_result("https://example.com/text", text="hello world", raw_bytes=None)
    errors = persist_downloads([result], tmp_path, allow_junk=True)
    assert errors["https://example.com/text"] is None
    blob_path = result.metadata.get("blob_path")
    assert blob_path
    with open(blob_path, "rb") as fh:
        assert fh.read() == b"hello world"


def test_persist_downloads_marks_missing_payload(tmp_path):
    result = _make_result("https://example.com/empty", text="", raw_bytes=None)
    errors = persist_downloads([result], tmp_path, allow_junk=True)
    assert errors["https://example.com/empty"] is None
    assert result.metadata.get("download_missing_no_payload") is True
