from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .fetcher_config import OVERRIDES_PATH
from .fetcher_utils import collect_environment_warnings


_SECRET_TOKENS = ("key", "token", "secret", "password", "pass")


def _is_secret_name(name: str) -> bool:
    lowered = (name or "").lower()
    return any(token in lowered for token in _SECRET_TOKENS)


def redact_value(value: str, keep: int = 4) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if len(raw) <= keep * 2:
        return "*" * len(raw)
    return f"{raw[:keep]}...{raw[-keep:]}"


def _redacted_env_value(name: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return redact_value(value) if _is_secret_name(name) else value


def _env_present(*names: str) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def _check_playwright_available() -> bool:
    try:
        from . import web_fetch
        return getattr(web_fetch, "async_playwright", None) is not None
    except Exception:
        return False


def _check_writable(path: Path) -> bool:
    try:
        if path.exists():
            return os.access(path, os.W_OK)
        parent = path.parent
        if not parent.exists():
            return False
        return os.access(parent, os.W_OK)
    except Exception:
        return False


def build_doctor_report(*, overrides_path: Optional[Path] = None) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "ok": True,
        "checks": [],
        "environment_warnings": collect_environment_warnings(),
    }

    def add_check(
        name: str,
        status: bool,
        *,
        detail: Optional[str] = None,
        remedy: Optional[str] = None,
        level: str = "warn",
        value: Optional[str] = None,
    ) -> None:
        entry = {
            "name": name,
            "status": "ok" if status else "missing",
            "level": level,
            "detail": detail,
        }
        if remedy:
            entry["remedy"] = remedy
        if value is not None:
            entry["value"] = _redacted_env_value(name, value)
        report["checks"].append(entry)
        if not status and level == "warn":
            report["ok"] = False

    brave_key = _env_present("BRAVE_API_KEY", "BRAVE_SEARCH_API_KEY")
    add_check(
        "BRAVE_API_KEY",
        bool(brave_key),
        detail="Brave alternates enabled" if brave_key else "Brave alternates disabled",
        remedy="Set BRAVE_API_KEY (or BRAVE_SEARCH_API_KEY) to enable alternate discovery.",
        level="warn",
        value=brave_key,
    )

    playwright_ok = _check_playwright_available()
    add_check(
        "playwright",
        playwright_ok,
        detail="SPA fallback enabled" if playwright_ok else "SPA fallback disabled",
        remedy="Install Playwright and run `playwright install --with-deps chromium`.",
        level="warn",
    )

    chutes_base = os.getenv("CHUTES_API_BASE")
    chutes_key = os.getenv("CHUTES_API_KEY")
    chutes_model = os.getenv("CHUTES_TEXT_MODEL")
    chutes_any = any([chutes_base, chutes_key, chutes_model])
    chutes_all = all([chutes_base, chutes_key, chutes_model])
    if chutes_any and not chutes_all:
        add_check(
            "CHUTES_*",
            False,
            detail="Partial SciLLM config; alternates may fail",
            remedy="Set CHUTES_API_BASE, CHUTES_API_KEY, CHUTES_TEXT_MODEL together.",
            level="warn",
        )
    elif chutes_all:
        add_check("CHUTES_*", True, detail="SciLLM alternates configured", level="info")
    else:
        add_check("CHUTES_*", False, detail="SciLLM alternates not configured", level="info")

    overrides = Path(os.getenv("FETCHER_OVERRIDES_PATH") or overrides_path or OVERRIDES_PATH)
    overrides_exists = overrides.exists()
    add_check(
        "FETCHER_OVERRIDES_PATH",
        overrides_exists,
        detail=str(overrides),
        remedy="Ensure overrides.json is present or set FETCHER_OVERRIDES_PATH.",
        level="warn",
    )

    cache_disabled = os.getenv("FETCHER_HTTP_CACHE_DISABLE", "0") == "1"
    if cache_disabled:
        add_check("FETCHER_HTTP_CACHE_DISABLE", True, detail="HTTP cache disabled", level="info")
    else:
        cache_path = Path(os.getenv("FETCHER_HTTP_CACHE_PATH") or (Path(os.getenv("FETCHER_HTTP_CACHE_DIR", "run/fetch_cache")) / "http_cache.json"))
        writable = _check_writable(cache_path)
        add_check(
            "FETCHER_HTTP_CACHE_PATH",
            writable,
            detail=str(cache_path),
            remedy="Create the cache directory or set FETCHER_HTTP_CACHE_PATH to a writable location.",
            level="warn",
        )

    gh_token = _env_present("GITHUB_TOKEN", "GH_TOKEN")
    add_check(
        "GITHUB_TOKEN",
        bool(gh_token),
        detail="GitHub auth available" if gh_token else "GitHub auth missing",
        remedy="Set GITHUB_TOKEN to reduce GitHub rate limits.",
        level="info",
        value=gh_token,
    )

    return report


def format_doctor_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("Fetcher doctor")
    lines.append(f"Generated: {report.get('generated_at')}")
    lines.append("Values are redacted where applicable.")
    lines.append("")
    for check in report.get("checks", []):
        name = check.get("name", "check")
        status = check.get("status", "unknown")
        level = check.get("level", "info")
        detail = check.get("detail")
        value = check.get("value")
        label = f"{name}: {status}"
        if value:
            label = f"{label} ({value})"
        lines.append(f"- [{level}] {label}")
        if detail:
            lines.append(f"  detail: {detail}")
        remedy = check.get("remedy")
        if remedy:
            lines.append(f"  remedy: {remedy}")
    warnings = report.get("environment_warnings") or []
    if warnings:
        lines.append("")
        lines.append("Environment warnings:")
        for warning in warnings:
            code = warning.get("code", "warning")
            message = warning.get("message", "")
            remedy = warning.get("remedy", "")
            lines.append(f"- {code}: {message}")
            if remedy:
                lines.append(f"  remedy: {remedy}")
    return "\n".join(lines).rstrip() + "\n"

