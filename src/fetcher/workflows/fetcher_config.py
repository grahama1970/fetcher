"""Fetcher defaults (endpoints, headers, domains, status codes, paths).

Centralizes static defaults so fetcher.py has no embedded magic strings.
These are baseline constants used to construct a policy; callers can inject
their own FetcherPolicy to override any of them.
"""

from __future__ import annotations

from pathlib import Path

# Endpoints / headers
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
HDR_ACCEPT = "Accept"
HDR_BRAVE_TOKEN = "X-Subscription-Token"

# Paths (project-relative)
_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SOURCES_DIR = _ROOT / "data" / "local_sources"
OVERRIDES_PATH = _ROOT / "data" / "overrides.json"

# Policy defaults
PAYWALL_DOMAINS = {
    "www.washingtonpost.com",
    "www.reuters.com",
    "www.bloomberg.com",
    "www.nytimes.com",
    "online.wsj.com",
    "www.wsj.com",
    "www.telegraph.co.uk",
    "money.cnn.com",
    "www.zdnet.com",
    "www.sentinelone.com",
    "www.arabsat.com",
    "www.almaobservatory.org",
    "spaceref.com",
    "www.spaceref.com",
    "spoonfeedin.blogspot.com.au",
    "ecadforum.com",
    "policy.defense.gov",
    "www.doctrine.af.mil",
    "www.dailynews.lk",
    "www.engr.utexas.edu",
    "www.eu-space.eu",
    "www.fortinet.com",
    "www.thuraya.com",
    "www.wired.com",
    "asec.ahnlab.com",
    "atlas.mitre.org",
    "securityboulevard.com",
    "irp.fas.org",
    "twitter.com",
    "x.com",
    "web.archive.org",
    "cheatsheetseries.owasp.org",
    "www.gps.gov",
    # Deterministic resolvers for public/government sources that frequently move
    "www.dtic.mil",
    "dtic.mil",
    "www.nasa.gov",
    "nasa.gov",
    "oig.nasa.gov",
    "www.c4i.org",
    "www.raeng.org.uk",
    "smcit-scc.space",
    "waterfall-security.com",
}

PAYWALL_STATUS_CODES = {401, 402, 403, 404, 500, 502, 503, 202, -1}
PAYWALL_HINTS = (
    "subscription",
    "subscribe",
    "paywall",
    "login",
    "sign in",
    "sign-in",
    "account",
    "purchase",
    "copyright",
    "members only",
)

BRAVE_EXCLUDED_DOMAINS = {
    "sparta.aerospace.org",
    "wikipedia.org",
    "en.wikipedia.org",
    "wikimedia.org",
    "twitter.com",
    "x.com",
}

