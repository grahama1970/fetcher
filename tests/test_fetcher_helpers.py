import json
from pathlib import Path

from fetcher.workflows import fetcher, paywall_detector, paywall_utils
from fetcher.workflows.fetcher_utils import resolve_repo_root
from fetcher.workflows.download_utils import (
    apply_download_mode,
    maybe_externalize_text,
)
from fetcher.workflows.web_fetch import FetchResult, _select_wayback_timestamp, _domain_matches_allowlist


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
