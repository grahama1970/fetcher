from fetcher.workflows import fetcher_utils


def test_collect_environment_warnings_brave_missing(monkeypatch):
    monkeypatch.delenv("BRAVE_API_KEY", raising=False)
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    warnings = fetcher_utils.collect_environment_warnings()
    codes = {item.get("code") for item in warnings}
    assert "brave_api_key_missing" in codes


def test_collect_environment_warnings_playwright_missing(monkeypatch):
    from fetcher.workflows import web_fetch

    monkeypatch.setattr(web_fetch, "async_playwright", None, raising=False)
    warnings = fetcher_utils.collect_environment_warnings()
    codes = {item.get("code") for item in warnings}
    assert "playwright_missing" in codes
