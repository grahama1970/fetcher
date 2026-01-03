import json
from datetime import datetime, timezone

from fetcher.workflows import fetcher as fetcher_module
from fetcher.workflows.web_fetch import FetchResult


def _make_result(url: str, *, status: int, text: str, fetch_path=None, content_verdict="ok", paywall_verdict="safe", extra=None):
    metadata = {"content_verdict": content_verdict, "paywall_verdict": paywall_verdict}
    if fetch_path is not None:
        metadata["fetch_path"] = fetch_path
    if extra:
        metadata.update(extra)
    return FetchResult(
        url=url,
        domain="example.com",
        status=status,
        content_type="text/html",
        text=text,
        fetched_at=datetime.now(timezone.utc).isoformat(),
        method="aiohttp",
        metadata=metadata,
    )


def test_compute_metrics_counts():
    r1 = _make_result("https://example.com/a", status=200, text="ok", fetch_path=["aiohttp"])
    r2 = _make_result(
        "https://example.com/b",
        status=403,
        text="",
        fetch_path=["aiohttp", "proxy_rotation", "wayback"],
        content_verdict="paywall",
        paywall_verdict="likely",
        extra={"proxy_rotation_attempted": True, "proxy_rotation_used": True},
    )
    metrics = fetcher_module._compute_metrics([r1, r2], audit={"rate_limit_metrics": {"total": 1}})
    assert metrics["counts"]["total"] == 2
    assert metrics["counts"]["success"] == 1
    assert metrics["counts"]["failed"] == 1
    assert metrics["content_verdict_counts"]["ok"] == 1
    assert metrics["content_verdict_counts"]["paywall"] == 1
    assert metrics["paywall_verdict_counts"]["safe"] == 1
    assert metrics["paywall_verdict_counts"]["likely"] == 1
    assert metrics["fallback_counts"]["proxy_rotation"] == 1
    assert metrics["fallback_counts"]["wayback"] == 1
    assert metrics["proxy_rotation"]["attempted"] == 1
    assert metrics["proxy_rotation"]["success"] == 1
    assert metrics["rate_limit_metrics"]["total"] == 1


def test_metrics_json_stdout_url_mode(monkeypatch, capsys):
    stub = _make_result("https://example.com", status=200, text="ok", fetch_path=["aiohttp", "wayback"])

    def fake_fetch_url(*args, **kwargs):
        return stub

    monkeypatch.setattr(fetcher_module, "fetch_url", fake_fetch_url)
    code = fetcher_module.main(["--url", "https://example.com", "--metrics-json", "-"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert "counts" in payload
    assert payload["counts"]["total"] == 1
