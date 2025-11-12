from pathlib import Path

from fetcher.workflows import fetcher
from fetcher.workflows.web_fetch import FetchResult, _select_wayback_timestamp


def test_resolve_repo_root_finds_data_dir(tmp_path: Path) -> None:
    repo_root = tmp_path / "project"
    data_dir = repo_root / "data" / "processed"
    data_dir.mkdir(parents=True)
    inventory = repo_root / "runs" / "inventory.jsonl"
    inventory.parent.mkdir(parents=True)
    inventory.write_text("{}\n", encoding="utf-8")

    resolved = fetcher._resolve_repo_root(inventory)

    assert resolved == repo_root


def test_resolve_repo_root_falls_back(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text("{}\n", encoding="utf-8")

    resolved = fetcher._resolve_repo_root(inventory)

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

    fetcher._maybe_externalize_text([result], tmp_path, max_inline_bytes=8)

    assert result.text == ""
    metadata = result.metadata or {}
    assert metadata.get("text_externalized") is True
    text_path = metadata.get("text_path")
    assert text_path is not None
    assert Path(text_path).exists()


def test_select_wayback_timestamp_prefers_first_column() -> None:
    payload = [
        ["timestamp", "original"],
        ["20240101010101", "https://example.com", "text/html", "200", "abc", "123"],
    ]

    assert _select_wayback_timestamp(payload) == "20240101010101"


def test_select_wayback_timestamp_handles_missing_rows() -> None:
    assert _select_wayback_timestamp([["timestamp"]]) is None
