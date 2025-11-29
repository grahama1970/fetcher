import json
from dataclasses import replace
import fetcher.workflows.alternate_urls as alternate_urls
from fetcher.workflows import paywall_utils
from fetcher.workflows.fetcher import DEFAULT_POLICY
from fetcher.workflows.web_fetch import FetchConfig, FetchResult


def _paywalled_result(url: str) -> FetchResult:
    return FetchResult(
        url=url,
        domain="example.com",
        status=403,
        content_type="text/html",
        text="",
        fetched_at="now",
        method="aiohttp",
        metadata={"paywall_verdict": "likely"},
    )


def _archived_result(url: str) -> FetchResult:
    return FetchResult(
        url=url,
        domain="web.archive.org",
        status=200,
        content_type="text/html",
        text="archived",
        fetched_at="now",
        method="aiohttp",
        metadata={},
    )


def test_wayback_precedes_perplexity(monkeypatch, tmp_path):
    base_domains = set(DEFAULT_POLICY.paywall_domains)
    base_domains.add("example.com")
    policy = replace(
        DEFAULT_POLICY,
        enable_perplexity_resolver=True,
        paywall_domains=frozenset(base_domains),
    )
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test-key")

    entry = {
        "url": "https://example.com/paywalled",
        "domain": "example.com",
        "controls": ["CTRL-1"],
    }
    entries = [dict(entry)]
    inventory_path = tmp_path / "inventory.jsonl"
    inventory_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    run_artifacts_dir = tmp_path / "artifacts"
    run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    results = [_paywalled_result(entry["url"])]

    flags = {"wayback": False, "perplexity": False}

    def fake_wayback(pending, *, limit, config, policy, run_fetch_loop, results, url_updates):
        flags["wayback"] = True
        for original_url in list(pending.keys()):
            pending.pop(original_url, None)
        return [
            {
                "control_id": "CTRL-1",
                "old_url": entry["url"],
                "new_url": "https://web.archive.org/web/20240101010101/https://example.com/paywalled",
                "provider": "wayback",
                "summary": "archive snapshot",
            }
        ]

    def fake_generate(*args, **kwargs):
        flags["perplexity"] = True
        return [], []

    def fake_validate(url, config, policy, run_fetch_loop):
        return _archived_result(url)

    monkeypatch.setattr(paywall_utils, "_apply_wayback_fallback", fake_wayback)
    monkeypatch.setattr(paywall_utils, "_search_brave_for_alternates", lambda *args, **kwargs: {})
    monkeypatch.setattr(paywall_utils, "_direct_override_candidate", lambda *args, **kwargs: None)
    monkeypatch.setattr(paywall_utils, "_validate_candidate_url", fake_validate)
    monkeypatch.setattr(alternate_urls, "generate_alternate_urls", fake_generate)

    root = tmp_path / "root"
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)

    applied = paywall_utils.resolve_paywalled_entries(
        entries=entries,
        results=results,
        config=FetchConfig(),
        root=root,
        inventory_path=inventory_path,
        run_artifacts_dir=run_artifacts_dir,
        limit=5,
        policy=policy,
        run_fetch_loop=lambda coro: ([], {}),
    )

    assert flags["wayback"] is True
    assert flags["perplexity"] is False
    assert applied and applied[0]["provider"] == "wayback"
