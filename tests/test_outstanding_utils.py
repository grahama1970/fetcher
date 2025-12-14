from dataclasses import dataclass

from fetcher.workflows.outstanding_utils import generate_outstanding_summary
from fetcher.workflows.web_fetch import FetchResult


@dataclass
class DummyPolicy:
    paywall_safe_domains: set
    paywall_safe_suffixes: tuple
    strip_subdomains: frozenset
    paywall_status_codes: set


def _result(
    url: str,
    domain: str,
    status: int,
    verdict: str | None = None,
    content_verdict: str | None = None,
    metadata_extra: dict | None = None,
) -> FetchResult:
    metadata: dict = {}
    if verdict:
        metadata["paywall_verdict"] = verdict
    if content_verdict:
        metadata["content_verdict"] = content_verdict
    if metadata_extra:
        metadata.update(metadata_extra)
    return FetchResult(
        url=url,
        domain=domain,
        status=status,
        content_type="text/html",
        text="",
        fetched_at="now",
        method="aiohttp",
        metadata=metadata,
        from_cache=False,
        raw_bytes=None,
    )


def test_generate_outstanding_summary_categories():
    policy = DummyPolicy(
        paywall_safe_domains={"example.org"},
        paywall_safe_suffixes=(".gov",),
        strip_subdomains=frozenset({"www"}),
        paywall_status_codes={402, 451},
    )
    url_rollup = [
        {"url": "https://example.org/doc", "domain": "example.org", "status": 200, "control_count": 1, "worksheets": []},
        {"url": "https://www.cisa.gov/alert", "domain": "www.cisa.gov", "status": 403, "control_count": 1, "worksheets": []},
        {"url": "https://paywall.example.com/post", "domain": "paywall.example.com", "status": 451, "control_count": 1, "worksheets": []},
        {"url": "https://missing.example.com/post", "domain": "missing.example.com", "status": 404, "control_count": 1, "worksheets": []},
        {
            "url": "https://www.cyberdefensemagazine.com/bot",
            "domain": "www.cyberdefensemagazine.com",
            "status": 403,
            "control_count": 1,
            "worksheets": [],
        },
        {
            "url": "https://secure.example.net/protected.pdf",
            "domain": "secure.example.net",
            "status": 200,
            "control_count": 1,
            "worksheets": [],
        },
        {
            "url": "https://cm.scholasticahq.com/article/5906.pdf",
            "domain": "cm.scholasticahq.com",
            "status": 200,
            "control_count": 1,
            "worksheets": [],
        },
    ]
    results_map = {
        "https://example.org/doc": _result("https://example.org/doc", "example.org", 200, verdict=None),
        "https://www.cisa.gov/alert": _result("https://www.cisa.gov/alert", "www.cisa.gov", 403, verdict=None),
        "https://paywall.example.com/post": _result(
            "https://paywall.example.com/post",
            "paywall.example.com",
            451,
            verdict="likely",
        ),
        "https://missing.example.com/post": _result("https://missing.example.com/post", "missing.example.com", 404, verdict=None),
        "https://www.cyberdefensemagazine.com/bot": _result(
            "https://www.cyberdefensemagazine.com/bot",
            "www.cyberdefensemagazine.com",
            403,
            verdict=None,
            content_verdict="thin",
            metadata_extra={
                "hub_title": "Just a moment...",
                "content_excerpt": "<html>Just a moment... Cloudflare is checking your browser...</html>",
            },
        ),
        "https://secure.example.net/protected.pdf": _result(
            "https://secure.example.net/protected.pdf",
            "secure.example.net",
            200,
            verdict=None,
            content_verdict="password_protected",
        ),
        "https://cm.scholasticahq.com/article/5906.pdf": _result(
            "https://cm.scholasticahq.com/article/5906.pdf",
            "cm.scholasticahq.com",
            200,
            verdict=None,
            content_verdict="missing_file",
        ),
    }

    summary = generate_outstanding_summary(url_rollup, results_map, policy)
    counts = summary["counts_by_category"]
    assert counts["needs_whitelist"] == 1
    assert counts["needs_login_or_playwright"] == 1
    assert counts["paywall"] == 1
    assert counts["broken_or_moved"] == 2
    # New bot-blocked category should be populated for Cloudflare-style interstitials.
    assert counts["bot_blocked"] == 1
    assert counts["password_protected"] == 1
