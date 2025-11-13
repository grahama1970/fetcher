"""Helper utilities for handling GitHub URLs inside the fetcher."""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse


@dataclass
class GithubRequest:
    """Encapsulates how a GitHub URL should be fetched."""

    original_url: str
    normalized_url: str
    fetch_url: str
    domain: str
    headers: Dict[str, str]
    metadata: Dict[str, Any]
    owner: Optional[str] = None
    repo: Optional[str] = None
    ref: Optional[str] = None
    path: Optional[str] = None
    allow_cli: bool = False


def _auth_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def prepare_github_request(original_url: str, normalized_url: str) -> Optional[GithubRequest]:
    """Return a GithubRequest when the URL points at a raw/blob resource."""

    parsed = urlparse(normalized_url)
    host = parsed.netloc.lower()
    path = parsed.path.strip("/")
    if not path:
        return None

    headers = _auth_headers()
    metadata: Dict[str, Any] = {"github_original": original_url}
    allow_cli = os.getenv("FETCHER_ENABLE_GH", "1") != "0"

    if host == "raw.githubusercontent.com":
        parts = path.split("/", 3)
        if len(parts) < 4:
            return None
        owner, repo, ref, remainder = parts
        metadata.update({
            "github_owner": owner,
            "github_repo": repo,
            "github_ref": ref,
            "github_path": remainder,
            "github_fetch": "raw",
        })
        return GithubRequest(
            original_url=original_url,
            normalized_url=normalized_url,
            fetch_url=normalized_url,
            domain=host,
            headers=headers,
            metadata=metadata,
            owner=owner,
            repo=repo,
            ref=ref,
            path=remainder,
            allow_cli=allow_cli,
        )

    if host in {"github.com", "www.github.com"}:
        parts = path.split("/")
        if len(parts) < 5:
            return None
        owner, repo, directive, ref = parts[:4]
        if directive not in {"blob", "raw"}:
            return None
        remainder = "/".join(parts[4:])
        fetch_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{remainder}"
        metadata.update({
            "github_owner": owner,
            "github_repo": repo,
            "github_ref": ref,
            "github_path": remainder,
            "github_fetch": "raw_redirect",
        })
        return GithubRequest(
            original_url=original_url,
            normalized_url=normalized_url,
            fetch_url=fetch_url,
            domain="raw.githubusercontent.com",
            headers=headers,
            metadata=metadata,
            owner=owner,
            repo=repo,
            ref=ref,
            path=remainder,
            allow_cli=allow_cli,
        )
    return None


def gh_cli_available() -> bool:
    return shutil.which("gh") is not None


async def fetch_with_cli(request: GithubRequest) -> Optional[tuple[int, str, str, str, Dict[str, Any], Optional[bytes]]]:
    """Use the GitHub CLI to download a file when HTTP access fails."""

    if not request.allow_cli or not gh_cli_available():
        return None
    if not (request.owner and request.repo and request.path):
        return None

    api_path = f"repos/{request.owner}/{request.repo}/contents/{request.path}"
    cmd = [
        "gh",
        "api",
        api_path,
        "-H",
        "Accept: application/vnd.github.v3.raw",
    ]
    if request.ref:
        cmd.extend(["-f", f"ref={request.ref}"])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        return None

    body = stdout
    text = body.decode("utf-8", "ignore") if body else ""
    metadata = {**request.metadata, "github_cli": True}
    return 200, "application/octet-stream", text, "github_cli", metadata, body


__all__ = [
    "GithubRequest",
    "prepare_github_request",
    "fetch_with_cli",
    "gh_cli_available",
]
