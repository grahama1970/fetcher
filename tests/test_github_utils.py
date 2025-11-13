from fetcher.workflows import github_utils


def test_prepare_github_request_blob(monkeypatch):
    monkeypatch.setenv("FETCHER_ENABLE_GH", "0")
    url = "https://github.com/org/repo/blob/main/docs/file.txt"
    request = github_utils.prepare_github_request(url, url)
    assert request is not None
    assert request.fetch_url == "https://raw.githubusercontent.com/org/repo/main/docs/file.txt"
    assert request.domain == "raw.githubusercontent.com"
    assert request.owner == "org"
    assert request.repo == "repo"
    assert request.ref == "main"
    assert request.path == "docs/file.txt"


def test_prepare_github_request_raw(monkeypatch):
    monkeypatch.setenv("FETCHER_ENABLE_GH", "0")
    url = "https://raw.githubusercontent.com/org/repo/main/README.md"
    request = github_utils.prepare_github_request(url, url)
    assert request is not None
    assert request.fetch_url == url
    assert request.domain == "raw.githubusercontent.com"


def test_prepare_github_request_non_github(monkeypatch):
    monkeypatch.setenv("FETCHER_ENABLE_GH", "0")
    url = "https://example.com/data.txt"
    request = github_utils.prepare_github_request(url, url)
    assert request is None
