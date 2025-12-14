import asyncio

from fetcher.workflows.web_fetch import FetchConfig, URLFetcher


def test_proxy_rotation_env_defaults(monkeypatch):
    monkeypatch.setenv("SPARTA_STEP06_PROXY_HOST", "rotator.local")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_PORT", "9000")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_USER", "user1")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_PASSWORD", "pass1")
    # Ensure defaults kick in when no domains are provided
    monkeypatch.delenv("SPARTA_STEP06_PROXY_DOMAINS", raising=False)
    fetcher = URLFetcher(FetchConfig())

    settings = fetcher._proxy_rotation
    assert settings is not None
    assert settings.display_endpoint == "rotator.local:9000"
    # Default allowlist includes D3FEND hosts
    assert "d3fend.mitre.org" in settings.allowed_domains
    # Default status trigger should watch for 429s
    assert 429 in settings.trigger_statuses


def test_proxy_rotation_trigger_on_rate_limit(monkeypatch):
    monkeypatch.setenv("SPARTA_STEP06_PROXY_HOST", "rotator.local")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_PORT", "9000")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_USER", "user1")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_PASSWORD", "pass1")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_DOMAINS", "d3fend.mitre.org")

    fetcher = URLFetcher(FetchConfig())
    fetcher._proxy_audit = fetcher._init_proxy_audit()

    proxy_calls = {"count": 0}

    async def fake_fetch_with_retries(
        self,
        session,
        url,
        domain,
        extra_headers=None,
        proxy_settings=None,
        proxy_reason=None,
    ):
        if proxy_settings is None:
            return 429, "text/html", "rate limit exceeded", "aiohttp", {}, b""
        proxy_calls["count"] += 1
        extra_meta = {
            "proxy_rotation_used": True,
            "proxy_rotation_reason": proxy_reason,
            "proxy_rotation_provider": proxy_settings.provider,
            "proxy_rotation_endpoint": proxy_settings.display_endpoint,
        }
        return 200, "text/html", "<html>ok</html>", "aiohttp", extra_meta, b"ok"

    monkeypatch.setattr(URLFetcher, "_fetch_with_retries", fake_fetch_with_retries, raising=False)

    async def run_once():
        sem = asyncio.Semaphore(1)
        entry = {"url": "https://d3fend.mitre.org/technique"}
        return await fetcher._fetch_entry(entry, None, sem, {})

    result = asyncio.run(run_once())

    assert result.status == 200
    assert result.metadata.get("proxy_rotation_used") is True
    assert result.metadata.get("proxy_rotation_reason") in {"status_429", "hint_rate limit exceeded"}
    assert proxy_calls["count"] == 1
    assert fetcher._proxy_audit is not None
    assert fetcher._proxy_audit.get("attempted") == 1
    assert fetcher._proxy_audit.get("success") == 1
    assert fetcher._proxy_audit.get("domains", {}).get("d3fend.mitre.org") == 1
    assert not result.metadata.get("proxy_rotation_credit_exhausted")


def test_proxy_rotation_flags_credit_exhaustion(monkeypatch):
    monkeypatch.setenv("SPARTA_STEP06_PROXY_HOST", "rotator.local")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_PORT", "9000")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_USER", "user1")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_PASSWORD", "pass1")
    monkeypatch.setenv("SPARTA_STEP06_PROXY_DOMAINS", "d3fend.mitre.org")

    fetcher = URLFetcher(FetchConfig())
    fetcher._proxy_audit = fetcher._init_proxy_audit()

    async def fake_fetch_with_retries(
        self,
        session,
        url,
        domain,
        extra_headers=None,
        proxy_settings=None,
        proxy_reason=None,
    ):
        if proxy_settings is None:
            return 429, "text/html", "rate limit exceeded", "aiohttp", {}, b""
        return 407, "text/html", "Out of credits. Please top up.", "aiohttp", {}, b""

    monkeypatch.setattr(URLFetcher, "_fetch_with_retries", fake_fetch_with_retries, raising=False)

    async def run_once():
        sem = asyncio.Semaphore(1)
        entry = {"url": "https://d3fend.mitre.org/technique"}
        return await fetcher._fetch_entry(entry, None, sem, {})

    result = asyncio.run(run_once())

    assert result.status == 407
    assert result.metadata.get("proxy_rotation_credit_exhausted") is True
    assert "out of credit" in result.metadata.get("proxy_rotation_credit_reason", "")
    assert fetcher._proxy_audit is not None
    assert fetcher._proxy_audit.get("credit_exhausted") == 1
