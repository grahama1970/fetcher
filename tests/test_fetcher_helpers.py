import asyncio
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest

from fetcher.workflows import fetcher, paywall_detector, paywall_utils
from fetcher.workflows.fetcher import _annotate_content_changes, _write_change_feed
from fetcher.workflows.fetcher_utils import build_failure_summary, resolve_repo_root, validate_url
from fetcher.workflows.download_utils import (
    apply_download_mode,
    maybe_externalize_text,
    annotate_paywall_metadata,
)
from fetcher.workflows.web_fetch import (
    FetchConfig,
    FetchResult,
    URLFetcher,
    _normalize_url,
    _looks_like_github_pages_404,
    _detect_soft_404,
    _select_wayback_timestamp,
    _domain_matches_allowlist,
    _summarize_rate_limits,
)
from dataclasses import replace


def test_resolve_repo_root_finds_data_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "project"
    data_dir = repo_root / "data" / "processed"
    data_dir.mkdir(parents=True)
    inventory = repo_root / "runs" / "inventory.jsonl"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("{}\n", encoding="utf-8")

    resolved = resolve_repo_root(inventory)

    assert resolved == repo_root


def test_resolve_repo_root_falls_back(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("{}\n", encoding="utf-8")

    resolved = resolve_repo_root(inventory)

    assert resolved == inventory.parent


def test_maybe_externalize_text(tmp_path: Path) -> None:
    result = FetchResult(
        url="https://example.com",
        domain="example.com",
        status=200,
        content_type="text/html",
        text="A" * 32,
        fetched_at="2024-01-01T00:00:00Z",
        method="aiohttp",
    )

    maybe_externalize_text([result], tmp_path, max_inline_bytes=8)

    assert result.text == ""
    metadata = result.metadata or {}
    assert metadata.get("text_externalized") is True
    assert metadata.get("text_inline_missing") is True
    text_path = metadata.get("text_path")
    assert text_path is not None
    assert Path(text_path).exists()
    assert metadata.get("file_path") == text_path
    assert metadata.get("text_length_chars") == 32

    # Ensure FetchResult serializes original length even when inline text removed
    payload = result.to_dict()
    assert payload["text_length"] == 32


def test_maybe_externalize_text_shared_cache(tmp_path: Path, monkeypatch) -> None:
    cache_dir = tmp_path / "shared"
    monkeypatch.setenv("FETCHER_TEXT_CACHE_DIR", str(cache_dir))
    result = FetchResult(
        url="https://example.com/shared",
        domain="example.com",
        status=200,
        content_type="text/html",
        text="cache-me",
        fetched_at="now",
        method="aiohttp",
    )

    run_dir = tmp_path / "run"
    maybe_externalize_text([result], run_dir, max_inline_bytes=0)

    metadata = result.metadata or {}
    cache_path = cache_dir / f"{metadata['text_sha256']}.txt"
    assert cache_path.exists()
    assert metadata.get("file_path") == str(cache_path)


def test_select_wayback_timestamp_prefers_first_column() -> None:
    payload = [
        ["timestamp", "original"],
        ["20240101010101", "https://example.com", "text/html", "200", "abc", "123"],
    ]

    assert _select_wayback_timestamp(payload) == "20240101010101"


def test_select_wayback_timestamp_handles_missing_rows() -> None:
    assert _select_wayback_timestamp([["timestamp"]]) is None


def test_domain_matches_allowlist() -> None:
    allow = {"mitre.org"}
    assert _domain_matches_allowlist("attack.mitre.org", allow) == "mitre.org"
    assert _domain_matches_allowlist("example.com", allow) is None


def test_analyze_link_hub_requires_low_body_or_high_density() -> None:
    fetcher = URLFetcher(FetchConfig(concurrency=1, per_domain=1))

    long_body = "word " * 500  # > 1200 chars
    many_links = "".join([f"<a href='/x{i}'>L{i}</a>" for i in range(25)])
    html = f"<html><head><title>T</title></head><body><p>{long_body}</p>{many_links}</body></html>"

    meta = fetcher._analyze_link_hub(html, "https://d3fend.mitre.org/tactic/d3f:Evict/")
    assert meta.get("link_hub") is False
    assert meta.get("link_hub_count", 0) >= 20

    small_body = "tiny"
    html2 = f"<html><head><title>T</title></head><body><p>{small_body}</p>{many_links}</body></html>"
    meta2 = fetcher._analyze_link_hub(html2, "https://d3fend.mitre.org/tactic/d3f:Evict/")
    assert meta2.get("link_hub") is True


def test_fetch_many_respects_config_toc_fanout_toggle(monkeypatch) -> None:
    calls = {"count": 0}

    async def fake_fetch_entry(self, entry, session, semaphore, per_domain):  # noqa: ANN001
        calls["count"] += 1
        url = entry["url"]
        if url.endswith("/hub/"):
            return FetchResult(
                url=url,
                domain="d3fend.mitre.org",
                status=200,
                content_type="text/html",
                text="<html><body>hub</body></html>",
                fetched_at="now",
                method="aiohttp",
                metadata={
                    "link_hub": True,
                    "link_hub_links": [
                        "https://d3fend.mitre.org/technique/d3f:ChildOne/",
                        "https://d3fend.mitre.org/technique/d3f:ChildTwo/",
                    ],
                },
            )
        body = ("para " * 250) + "\n\n" + ("para " * 250) + "\n\n" + ("para " * 250)
        return FetchResult(
            url=url,
            domain="d3fend.mitre.org",
            status=200,
            content_type="text/plain",
            text=body,
            fetched_at="now",
            method="aiohttp",
            metadata={"source": entry.get("source")},
        )

    monkeypatch.setattr(URLFetcher, "_fetch_entry", fake_fetch_entry, raising=False)

    monkeypatch.setenv("SPARTA_TOC_FANOUT", "1")
    disabled = URLFetcher(FetchConfig(concurrency=2, per_domain=2, enable_toc_fanout=False))
    results, _ = asyncio.run(disabled.fetch_many([{"url": "https://d3fend.mitre.org/hub/"}]))
    assert calls["count"] == 1
    assert len(results) == 1

    calls["count"] = 0
    monkeypatch.setenv("SPARTA_TOC_FANOUT", "0")
    enabled = URLFetcher(FetchConfig(concurrency=2, per_domain=2, enable_toc_fanout=True))
    results2, _ = asyncio.run(enabled.fetch_many([{"url": "https://d3fend.mitre.org/hub/"}]))
    assert calls["count"] == 3
    assert len(results2) == 3


def test_content_change_tracker_detects_delta(tmp_path: Path) -> None:
    cache_path = tmp_path / "hashes.json"
    r1 = FetchResult(
        url="https://example.com/resource",
        domain="example.com",
        status=200,
        content_type="text/html",
        text="alpha",
        fetched_at="t1",
        method="aiohttp",
        metadata={"url_normalized": "https://example.com/resource"},
    )
    _annotate_content_changes([r1], cache_path)
    assert cache_path.exists()
    assert r1.metadata.get("content_changed") is True

    r2 = FetchResult(
        url="https://example.com/resource",
        domain="example.com",
        status=200,
        content_type="text/html",
        text="alpha",
        fetched_at="t2",
        method="aiohttp",
        metadata={"url_normalized": "https://example.com/resource"},
    )
    _annotate_content_changes([r2], cache_path)
    assert r2.metadata.get("content_changed") is False
    assert r2.metadata.get("content_previous_sha256") == r1.metadata.get("content_sha256")

    r3 = FetchResult(
        url="https://example.com/resource",
        domain="example.com",
        status=200,
        content_type="text/html",
        text="alpha beta",
        fetched_at="t3",
        method="aiohttp",
        metadata={"url_normalized": "https://example.com/resource"},
    )
    _annotate_content_changes([r3], cache_path)
    assert r3.metadata.get("content_changed") is True
    assert r3.metadata.get("content_previous_sha256") == r2.metadata.get("content_sha256")


def test_write_change_feed(tmp_path: Path) -> None:
    run_dir = tmp_path / "artifacts"
    results = [
        FetchResult(
            url="https://example.com/one",
            domain="example.com",
            status=200,
            content_type="text/html",
            text="",
            fetched_at="today",
            method="aiohttp",
            metadata={
                "content_changed": True,
                "content_sha256": "abc",
                "content_previous_sha256": "def",
                "content_previous_fetched_at": "yesterday",
            },
        ),
        FetchResult(
            url="https://example.com/two",
            domain="example.com",
            status=200,
            content_type="text/html",
            text="",
            fetched_at="today",
            method="aiohttp",
            metadata={"content_changed": False},
        ),
    ]
    _write_change_feed(results, run_dir)
    changes_path = run_dir / "changes.jsonl"
    assert changes_path.exists()
    lines = changes_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["url"] == "https://example.com/one"
    assert payload["content_previous_sha256"] == "def"


def test_summarize_rate_limits_helper() -> None:
    results = [
        FetchResult(
            url="https://a.example",
            domain="a.example",
            status=200,
            content_type="text/html",
            text="",
            fetched_at="now",
            method="aiohttp",
        ),
        FetchResult(
            url="https://a.example/limit",
            domain="a.example",
            status=429,
            content_type="text/html",
            text="",
            fetched_at="now",
            method="aiohttp",
        ),
        FetchResult(
            url="https://b.example",
            domain="b.example",
            status=200,
            content_type="text/html",
            text="",
            fetched_at="now",
            method="aiohttp",
        ),
    ]
    metrics = _summarize_rate_limits(results, runtime_seconds=2.0)
    per_domain = metrics["per_domain"]
    assert per_domain["a.example"]["requests"] == 2
    assert per_domain["a.example"]["rate_limited"] == 1
    assert per_domain["b.example"]["rate_limited"] == 0


def test_apply_download_mode_download_only(tmp_path: Path) -> None:
    result = FetchResult(
        url="https://example.com/alpha.pdf",
        domain="example.com",
        status=200,
        content_type="application/pdf",
        text="PDF CONTENT",
        fetched_at="2024-01-01T00:00:00Z",
        method="aiohttp",
        raw_bytes=b"PDF CONTENT",
    )

    artifacts = tmp_path / "artifacts"
    apply_download_mode([result], artifacts, "download_only", 1000, 500, 0)

    metadata = result.metadata or {}
    blob_path = Path(metadata["blob_path"])
    assert blob_path.exists()
    assert metadata["download_mode"] == "download_only"
    assert result.text == ""
    assert result.raw_bytes is None
    assert metadata.get("file_path") == str(blob_path)


def test_apply_download_mode_rolling_extract(tmp_path: Path) -> None:
    sentences = [
        "This is sentence one.",
        "Here is sentence two with more words.",
        "Sentence three ensures overlap across windows.",
        "Finally, sentence four closes the test case.",
    ]
    text = " ".join(sentences)
    result = FetchResult(
        url="https://example.com/doc.html",
        domain="example.com",
        status=200,
        content_type="text/html",
        text=text,
        fetched_at="2024-01-01T00:00:00Z",
        method="aiohttp",
        raw_bytes=text.encode("utf-8"),
    )

    artifacts = tmp_path / "artifacts"
    apply_download_mode([result], artifacts, "rolling_extract", 120, 60, 5)

    metadata = result.metadata or {}
    windows_path = Path(metadata["rolling_windows_path"])
    assert windows_path.exists()
    with windows_path.open("r", encoding="utf-8") as fh:
        windows = [json.loads(line) for line in fh]
    assert len(windows) == metadata["rolling_windows_count"]
    assert len(windows) <= 5
    assert len(windows) >= 2
    for window in windows:
        assert window["text"].strip()
        assert window["end"] > window["start"]
        # Windows should end at sentence boundaries ('.', '!' or '?')
        if window["end"] < len(text):
            assert text[window["end"] - 1] in ".!?"


def test_detect_paywall_flags_subscription_language() -> None:
    html = """
    <html><body>
        <p>This article is for subscribers only. Please subscribe to continue reading this premium article.</p>
    </body></html>
    """
    detection = paywall_detector.detect_paywall(
        url="https://example.com/paywalled",
        status=200,
        html=html,
    )
    assert detection["verdict"] in {"maybe", "likely"}


def test_detect_paywall_respects_safe_domains() -> None:
    html = """
    <html><body><div>Please sign in to continue reading.</div></body></html>
    """

    class Policy:
        paywall_safe_domains = {"attack.mitre.org"}

    detection = paywall_detector.detect_paywall(
        url="https://attack.mitre.org/mitigations/M1032/",
        status=200,
        html=html,
        policy=Policy(),
    )

    assert detection["verdict"] == "unlikely"
    assert detection["indicators"].get("safe_domain") == "attack.mitre.org"


def test_detect_paywall_respects_safe_suffixes() -> None:
    html = """
    <html><body><div>Account holders may subscribe for updates.</div></body></html>
    """

    class Policy:
        paywall_safe_domains = set()
        paywall_safe_suffixes = (".gov",)

    detection = paywall_detector.detect_paywall(
        url="https://www.cisa.gov/resources",
        status=200,
        html=html,
        policy=Policy(),
    )

    assert detection["verdict"] == "unlikely"
    assert detection["indicators"].get("safe_domain_suffix") in {".gov", "gov"}


def test_detect_paywall_marks_known_paywall_domains() -> None:
    html = "<html><body><div>Sample teaser only</div></body></html>"

    class Policy:
        paywall_domains = {"hbr.org"}
        paywall_safe_domains = set()

    detection = paywall_detector.detect_paywall(
        url="https://hbr.org/1964/01/profit-from-the-learning-curve",
        status=200,
        html=html,
        policy=Policy(),
    )

    assert detection["verdict"] in {"maybe", "likely"}
    assert detection["indicators"].get("paywall_domain") == "hbr.org"


def test_paywall_verdict_gate_respects_annotations() -> None:
    result = FetchResult(
        url="https://example.com",
        domain="example.com",
        status=403,
        content_type="text/html",
        text="",
        fetched_at="2024-01-01T00:00:00Z",
        method="aiohttp",
        metadata={"paywall_verdict": "unlikely"},
    )

    assert not paywall_utils._verdict_allows_resolver(result)

    result.metadata["paywall_verdict"] = "maybe"
    assert paywall_utils._verdict_allows_resolver(result)

    result.metadata = {}
    assert paywall_utils._verdict_allows_resolver(result)


def test_normalize_url_preserves_trailing_slash_and_casing() -> None:
    url = "https://d3fend.mitre.org/dao/artifact/d3f:OutboundInternetNetworkTraffic/"
    normalized = _normalize_url(url)
    assert normalized == url  # path + trailing slash preserved

    other = "https://example.com/Path/With/Slash/"
    normalized_other = _normalize_url(other)
    assert normalized_other == other


def test_normalize_url_strips_whitespace_in_path() -> None:
    messy = "https://d3fend.mitre.org/technique/ d3f:AssetVulnerabilityEnumeration/"
    assert _normalize_url(messy) == "https://d3fend.mitre.org/technique/d3f:AssetVulnerabilityEnumeration/"

    messy_other = "https://example.com/path/ with spaces /value"
    assert _normalize_url(messy_other) == "https://example.com/path/withspaces/value"


def test_github_pages_404_signature_detection() -> None:
    html = "<html><head><title>Page not found · GitHub Pages</title></head><body>The site configured at this address does not contain the requested file.</body></html>"
    assert _looks_like_github_pages_404(html)
    assert not _looks_like_github_pages_404("<html><body>Real content</body></html>")

def test_detect_soft_404_templates() -> None:
    gh = "<html><title>Page not found · GitHub Pages</title><body>The site configured at this address does not contain the requested file.</body></html>"
    cf = "<html><body><h1>Access denied | cloudflare</h1><p>Error 1020</p><p>You are unable to access example.com</p></body></html>"
    s3 = "<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message></Error>"
    generic = "<html><body><h1>Page not found</h1><p>The requested URL was not found on this server</p></body></html>"

    assert _detect_soft_404(gh) == "github_pages_404"
    assert _detect_soft_404(cf) == "cloudflare_access_denied"
    assert _detect_soft_404(s3) == "s3_nosuchkey"
    assert _detect_soft_404(generic) == "generic_not_found"
    assert _detect_soft_404("<html><body>Real content</body></html>") is None


@pytest.mark.asyncio
async def test_github_pages_404_marks_result_failure(monkeypatch) -> None:
    html = "<html><head><title>Page not found · GitHub Pages</title></head><body>The site configured at this address does not contain the requested file.</body></html>"

    monkeypatch.setattr("fetcher.workflows.web_fetch.async_playwright", None)
    fetcher = URLFetcher(FetchConfig())
    fetcher._local_data_root = None  # force network path, no local dataset shortcut
    monkeypatch.setattr(fetcher, "_fetch_d3fend_local", lambda *args, **kwargs: None, raising=False)

    class _StubResponse:
        def __init__(self, body: str):
            self.status = 200
            self._body = body.encode()
            self.headers = {"Content-Type": "text/html"}

        async def read(self):
            return self._body

    class _StubSession:
        def __init__(self, response: _StubResponse):
            self._response = response

        def get(self, *_args, **_kwargs):
            return self

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_fetch_from_wayback(_session, _url):
        return None

    async def fake_fetch_from_jina(_session, _url):
        return None

    async def via_fetch_once(_session, _url, _domain, **_kwargs):
        return await fetcher._fetch_once(_StubSession(_StubResponse(html)), _url, _domain, **_kwargs)

    fetcher._cache_enabled = False
    monkeypatch.setattr(fetcher, "_fetch_with_retries", via_fetch_once)
    monkeypatch.setattr(fetcher, "_fetch_from_wayback", fake_fetch_from_wayback)
    monkeypatch.setattr(fetcher, "_fetch_from_jina", fake_fetch_from_jina)
    monkeypatch.setattr(fetcher, "_needs_playwright", lambda *args, **kwargs: False)
    monkeypatch.setattr("fetcher.workflows.web_fetch._detect_soft_404", lambda _text: "github_pages_404")

    entry = {
        "url": "https://d3fend.mitre.org/dao/artifact/d3f:OutboundInternetNetworkTraffic/",
        "domain": "d3fend.mitre.org",
        "controls": [],
        "titles": [],
        "frameworks": [],
        "worksheets": [],
    }

    result = await fetcher._fetch_entry(entry, session=None, semaphore=asyncio.Semaphore(10), per_domain={})

    assert result.status == 404
    assert result.text == ""
    assert result.metadata.get("soft_404_detected") is True
    assert result.metadata.get("soft_404_template") == "github_pages_404"


def test_annotate_paywall_runs_on_empty_text_for_paywall_domain() -> None:
    result = FetchResult(
        url="https://hbr.org/paywalled-article",
        domain="hbr.org",
        status=200,
        content_type="text/html",
        text="",
        fetched_at="now",
        method="aiohttp",
    )

    annotate_paywall_metadata([result], fetcher.DEFAULT_POLICY)

    md = result.metadata or {}
    assert md.get("paywall_verdict") == "maybe"
    assert md.get("paywall_detection", {}).get("indicators", {}).get("no_text_body") is True


@pytest.mark.asyncio
async def test_resolver_considers_paywall_domain_with_empty_text(monkeypatch, tmp_path: Path) -> None:
    entry = {
        "url": "https://hbr.org/1964/01/profit-from-the-learning-curve",
        "domain": "hbr.org",
        "controls": [],
        "titles": [],
        "frameworks": [],
        "worksheets": [],
    }
    result = FetchResult(
        url=entry["url"],
        domain="hbr.org",
        status=200,
        content_type="text/html",
        text="",
        fetched_at="now",
        method="aiohttp",
    )
    annotate_paywall_metadata([result], fetcher.DEFAULT_POLICY)

    called = {"brave": False}

    def fake_search_brave(targets, contexts, config, policy, run_fetch_loop, debug_log_path=None):
        called["brave"] = True
        return {}

    monkeypatch.setattr(paywall_utils, "_search_brave_for_alternates", fake_search_brave)

    inventory_path = tmp_path / "inv.jsonl"
    inventory_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    applied = paywall_utils.resolve_paywalled_entries(
        entries=[entry],
        results=[result],
        config=FetchConfig(),
        root=tmp_path,
        inventory_path=inventory_path,
        run_artifacts_dir=tmp_path,
        limit=1,
        policy=fetcher.DEFAULT_POLICY,
        run_fetch_loop=lambda coro: ([], {}),
    )

    assert called["brave"] is True
    assert applied == []


@pytest.mark.asyncio
async def test_cache_skips_d3fend_local_when_local_root_disabled(monkeypatch, tmp_path: Path) -> None:
    config = FetchConfig(disable_http_cache=False)
    fetcher = URLFetcher(config, cache_path=None)

    url = "https://d3fend.mitre.org/dao/artifact/d3f:OutboundInternetNetworkTraffic/"
    url_norm = url
    fetcher._cache_enabled = True
    fetcher._cache[url_norm] = {
        "status": 200,
        "content_type": "text/plain",
        "text": "cached",
        "fetched_at": "now",
        "method": "d3fend_local",
        "metadata": {},
    }
    fetcher._local_data_root = None  # simulate disabled local data root

    async def fake_fetch_with_retries(_session, _url, _domain, **_kwargs):
        return 200, "text/plain", "live", "aiohttp", {}, b"live"

    monkeypatch.setattr(fetcher, "_fetch_with_retries", fake_fetch_with_retries)
    monkeypatch.setattr(fetcher, "_fetch_from_wayback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetcher, "_fetch_from_jina", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetcher, "_needs_playwright", lambda *args, **kwargs: False)

    entry = {
        "url": url,
        "domain": "d3fend.mitre.org",
        "controls": [],
        "titles": [],
        "frameworks": [],
        "worksheets": [],
    }

    result = await fetcher._fetch_entry(entry, session=None, semaphore=asyncio.Semaphore(5), per_domain={})

    assert result.text == "live"
    assert result.method == "aiohttp"
    assert result.from_cache is False


@pytest.mark.asyncio
async def test_disable_http_cache_flag_ignores_cache(monkeypatch, tmp_path: Path) -> None:
    cache_file = tmp_path / "cache.json"
    cache_file.write_text(
        json.dumps(
            {
                "https://example.com": {
                    "status": 200,
                    "content_type": "text/html",
                    "text": "cached-body",
                    "fetched_at": "now",
                    "method": "aiohttp",
                    "metadata": {},
                }
            }
        ),
        encoding="utf-8",
    )

    config = FetchConfig(disable_http_cache=True)
    fetcher = URLFetcher(config, cache_path=cache_file)

    async def fake_fetch_with_retries(_session, _url, _domain, **_kwargs):
        return 200, "text/html", "fresh-body", "aiohttp", {}, b"fresh-body"

    monkeypatch.setattr(fetcher, "_fetch_with_retries", fake_fetch_with_retries)
    monkeypatch.setattr(fetcher, "_fetch_from_wayback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetcher, "_fetch_from_jina", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetcher, "_needs_playwright", lambda *args, **kwargs: False)

    entry = {
        "url": "https://example.com/",
        "domain": "example.com",
        "controls": [],
        "titles": [],
        "frameworks": [],
        "worksheets": [],
    }

    result = await fetcher._fetch_entry(entry, session=None, semaphore=asyncio.Semaphore(5), per_domain={})
    assert result.text == "fresh-body"
    assert result.from_cache is False


@pytest.mark.asyncio
async def test_cache_bypassed_when_target_local_override(monkeypatch, tmp_path: Path) -> None:
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            [
                {
                    "domain": "example.com",
                    "path_prefix": "/paywalled.pdf",
                    "target_local_file": "mirror.pdf",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FETCHER_OVERRIDES_PATH", str(overrides))
    from fetcher.workflows import web_fetch
    web_fetch._TARGET_LOCAL_OVERRIDE_CACHE.clear()

    url = "https://example.com/paywalled.pdf"
    fetcher = URLFetcher(FetchConfig())
    fetcher._cache_enabled = True
    fetcher._cache[url] = {
        "status": 200,
        "content_type": "text/html",
        "text": "cached",
        "fetched_at": "now",
        "method": "aiohttp",
        "metadata": {},
    }

    async def fresh(_session, _url, _domain, **_kwargs):
        return 200, "text/html", "fresh-body", "aiohttp", {}, b"fresh-body"

    monkeypatch.setattr(fetcher, "_fetch_with_retries", fresh)
    monkeypatch.setattr(fetcher, "_fetch_from_wayback", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetcher, "_fetch_from_jina", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(fetcher, "_needs_playwright", lambda *args, **kwargs: False)

    entry = {
        "url": url,
        "domain": "example.com",
        "controls": [],
        "titles": [],
        "frameworks": [],
        "worksheets": [],
    }

    result = await fetcher._fetch_entry(entry, session=None, semaphore=asyncio.Semaphore(5), per_domain={})
    assert result.text == "fresh-body"
    assert result.from_cache is False


def test_direct_override_accepts_absolute_target_local_file(tmp_path: Path) -> None:
    local_file = tmp_path / "mirror.txt"
    local_file.write_text("mirror", encoding="utf-8")

    policy = replace(fetcher.DEFAULT_POLICY, overrides_path=tmp_path / "overrides.json", local_sources_dir=tmp_path / "missing")
    override_rule = {
        "domain": "example.com",
        "path_prefix": "/abs",
        "target_local_file": str(local_file),
    }
    policy.overrides_path.write_text(json.dumps([override_rule]), encoding="utf-8")

    entry = {"url": "https://example.com/abs/path"}
    candidate = paywall_utils._direct_override_candidate(entry, policy)
    assert candidate is not None
    uri, rule = candidate
    assert uri == local_file.as_uri()


def test_fetch_from_file_allows_external_local_sources(tmp_path: Path) -> None:
    external_dir = tmp_path / "devops" / "sparta" / "data" / "local_sources"
    external_dir.mkdir(parents=True)
    local_file = external_dir / "reuters_override.txt"
    local_file.write_text("external mirror", encoding="utf-8")

    fetcher = URLFetcher(FetchConfig())
    status, content_type, text, method, metadata, raw_bytes = fetcher._fetch_from_file(urlparse(local_file.as_uri()))

    assert status == 200
    assert content_type == "text/plain"
    assert text == "external mirror"
    assert method == "file"
    assert metadata.get("local_path") == str(local_file)
    assert raw_bytes == b"external mirror"


def test_validate_candidate_url_accepts_external_file(tmp_path: Path) -> None:
    local_file = tmp_path / "local_sources" / "override.txt"
    local_file.parent.mkdir(parents=True)
    local_file.write_text("override content", encoding="utf-8")

    config = FetchConfig()

    def run_fetch_loop(coro):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    result = paywall_utils._validate_candidate_url(
        local_file.as_uri(),
        config,
        fetcher.DEFAULT_POLICY,
        run_fetch_loop,
    )

    assert result is not None
    assert result.status == 200
    assert result.method == "file"
    assert result.text == "override content"
    assert result.metadata.get("local_path") == str(local_file)


def test_validate_url_rejects_invalid() -> None:
    assert validate_url("https://example.com") is None
    assert validate_url("http://") == "missing_host"
    assert validate_url("ftp://example.com") == "unsupported_scheme"
    assert validate_url("file://") == "missing_path"


def test_build_failure_summary_counts() -> None:
    ok = FetchResult(
        url="https://example.com/ok",
        domain="example.com",
        status=200,
        content_type="text/html",
        text="ok",
        fetched_at="now",
        method="aiohttp",
        metadata={"content_verdict": "ok", "fallback_reason": "none"},
    )
    fail = FetchResult(
        url="https://example.com/fail",
        domain="example.com",
        status=404,
        content_type="text/html",
        text="",
        fetched_at="now",
        method="aiohttp",
        metadata={"content_verdict": "junk", "fallback_reason": "soft_404"},
    )
    summary = build_failure_summary([ok, fail])
    assert summary["status_counts"]["200"] == 1
    assert summary["status_counts"]["404"] == 1
    assert summary["content_verdict_counts"]["ok"] == 1
    assert summary["content_verdict_counts"]["junk"] == 1
    assert summary["fallback_reason_counts"]["none"] == 1
    assert summary["fallback_reason_counts"]["soft_404"] == 1
