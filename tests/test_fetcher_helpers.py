import json
from pathlib import Path

from fetcher.workflows import fetcher, paywall_detector, paywall_utils
from fetcher.workflows.fetcher import _annotate_content_changes
from fetcher.workflows.fetcher_utils import resolve_repo_root
from fetcher.workflows.download_utils import (
    apply_download_mode,
    maybe_externalize_text,
)
from fetcher.workflows.web_fetch import (
    FetchResult,
    _select_wayback_timestamp,
    _domain_matches_allowlist,
    _summarize_rate_limits,
)


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
