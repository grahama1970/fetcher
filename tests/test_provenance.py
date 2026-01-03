from fetcher.workflows import web_fetch


def test_fetch_path_playwright():
    path = web_fetch._build_fetch_path("playwright", {"playwright": True}, from_cache=False)
    assert path == ["aiohttp", "playwright"]


def test_fetch_path_wayback():
    path = web_fetch._build_fetch_path("wayback", {"via": "wayback"}, from_cache=False)
    assert path == ["aiohttp", "wayback"]


def test_fetch_path_jina():
    path = web_fetch._build_fetch_path("jina", {"via": "jina"}, from_cache=False)
    assert path == ["aiohttp", "jina"]


def test_fetch_path_proxy_rotation():
    path = web_fetch._build_fetch_path("aiohttp", {"proxy_rotation_used": True}, from_cache=False)
    assert path == ["aiohttp", "proxy_rotation"]


def test_fetch_path_cache():
    path = web_fetch._build_fetch_path("aiohttp", {}, from_cache=True)
    assert path == ["cache"]
