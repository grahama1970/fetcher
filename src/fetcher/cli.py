from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Set

import typer

from .consumer import load_manifest, parse_emit_csv, run_consumer

app = typer.Typer(add_help_option=False, no_args_is_help=False)


def _minimal_help() -> str:
    return """Fetcher (consumer CLI)

Usage:
  fetcher get <url> [--out <DIR>] [--emit <CSV>] [--json] [--soft-fail]
  fetcher get-manifest <urls.txt|-> [--out <DIR>] [--emit <CSV>] [--json] [--soft-fail]

Common options:
  --out <DIR>     Write artifacts into this directory (no subdir).
  --emit <CSV>    Emit outputs: download,text,md,fit_md (default: download,text,md).
  --json          Print consumer_summary.json to stdout only.
  --soft-fail     Exit 0 even if some items fail.

Discoverability:
  --help-full     Expanded help + env vars + artifacts.
  --find <query>  Search commands, flags, env vars, artifacts.
"""


def _help_full() -> str:
    return """Fetcher consumer CLI (best-effort)

Commands:
  get            Fetch a single URL.
  get-manifest   Fetch URLs from a strict line-based manifest (file or stdin).

Emit toggles (--emit CSV):
  download  Persist raw bytes (default ON).
  text      Materialize extracted text.
  md        Materialize markdown.
  fit_md    Materialize fit-markdown (only when md enabled).

Artifacts (consumer):
  consumer_summary.json  Stable JSON summary for the run.
  Walkthrough.md         Deterministic walkthrough derived from summary.
  downloads/             Raw downloads (when download emitted).
  extracted_text/        Extracted text artifacts (when text emitted).
  markdown/              Markdown artifacts (when md emitted).
  fit_markdown/          Fit-markdown artifacts (when fit_md emitted).

Important env vars (existing):
  BRAVE_API_KEY
  FETCHER_HTTP_CACHE_DISABLE
  FETCHER_HTTP_CACHE_PATH
  FETCHER_OVERRIDES_PATH
  FETCHER_ENABLE_PDF_DISCOVERY
  FETCHER_PDF_DISCOVERY_MAX
  FETCHER_EMIT_EXTRACTED_TEXT
  FETCHER_EMIT_MARKDOWN
  FETCHER_EMIT_FIT_MARKDOWN

Troubleshooting:
  - If Playwright isn’t installed, browser fallbacks are skipped.
  - Use fetcher-etl for full ETL reports and knobs.
"""


_FIND_INDEX = [
    ("command", "get", "Fetch a single URL."),
    ("command", "get-manifest", "Fetch URLs from a manifest file or stdin."),
    ("flag", "--out", "Write artifacts into this directory (no subdir)."),
    ("flag", "--emit", "Emit outputs: download,text,md,fit_md."),
    ("flag", "--json", "Print summary JSON to stdout only."),
    ("flag", "--soft-fail", "Exit 0 even if some items fail."),
    ("flag", "--help-full", "Expanded help, env vars, artifacts."),
    ("flag", "--find", "Search commands, flags, env vars, artifacts."),
    ("env", "BRAVE_API_KEY", "Enable Brave alternates."),
    ("env", "FETCHER_HTTP_CACHE_DISABLE", "Disable HTTP cache read/write."),
    ("env", "FETCHER_HTTP_CACHE_PATH", "Override HTTP cache path."),
    ("env", "FETCHER_OVERRIDES_PATH", "Override overrides.json path."),
    ("env", "FETCHER_ENABLE_PDF_DISCOVERY", "Enable PDF discovery (best-effort)."),
    ("env", "FETCHER_PDF_DISCOVERY_MAX", "Max PDFs discovered per page."),
    ("artifact", "consumer_summary.json", "Stable summary output."),
    ("artifact", "Walkthrough.md", "Deterministic walkthrough."),
    ("artifact", "downloads/", "Raw downloads."),
    ("artifact", "extracted_text/", "Extracted text artifacts."),
    ("artifact", "markdown/", "Markdown artifacts."),
    ("artifact", "fit_markdown/", "Fit-markdown artifacts."),
]


def _run_find(query: str) -> str:
    needle = (query or "").strip().lower()
    if not needle:
        return ""
    lines = []
    for category, name, desc in _FIND_INDEX:
        haystack = f"{category} {name} {desc}".lower()
        if needle in haystack:
            lines.append(f"{category} {name} - {desc}")
    return "\n".join(lines)


def _parse_emit(value: str) -> Set[str]:
    allowed = {"download", "text", "md", "fit_md"}
    emit = parse_emit_csv(value)
    unknown = sorted(token for token in emit if token not in allowed)
    if unknown:
        raise typer.BadParameter(f"Unknown emit option(s): {', '.join(unknown)}")
    return emit or {"download", "text", "md"}


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    help: bool = typer.Option(False, "--help", "-h", is_eager=True, help="Show minimal help."),
    help_full: bool = typer.Option(False, "--help-full", is_eager=True, help="Show expanded help."),
    find: Optional[str] = typer.Option(None, "--find", is_eager=True, help="Search commands, flags, env vars, artifacts."),
) -> None:
    if help_full:
        typer.echo(_help_full())
        raise typer.Exit(code=0)
    if find is not None:
        output = _run_find(find)
        if output:
            typer.echo(output)
        raise typer.Exit(code=0)
    if help or ctx.invoked_subcommand is None:
        typer.echo(_minimal_help())
        raise typer.Exit(code=0)


@app.command("get", add_help_option=True)
def get_url(
    url: str = typer.Argument(..., help="URL to fetch."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write artifacts into this directory (no subdir)."),
    emit: str = typer.Option("download,text,md", "--emit", help="Emit outputs: download,text,md,fit_md."),
    json_out: bool = typer.Option(False, "--json", help="Print summary JSON to stdout only."),
    soft_fail: bool = typer.Option(False, "--soft-fail", help="Exit 0 even if some items fail."),
) -> None:
    emit_set = _parse_emit(emit)
    try:
        summary, exit_code = run_consumer(
            [url],
            command="get",
            out_dir=out,
            emit=emit_set,
            soft_fail=soft_fail,
        )
    except Exception as exc:
        if not json_out:
            typer.echo(f"fatal: {exc}", err=True)
        raise typer.Exit(code=3)
    if json_out:
        sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    raise typer.Exit(code=exit_code)


@app.command("get-manifest", add_help_option=True)
def get_manifest(
    path_or_dash: str = typer.Argument(..., help="Path to manifest or '-' for stdin."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write artifacts into this directory (no subdir)."),
    emit: str = typer.Option("download,text,md", "--emit", help="Emit outputs: download,text,md,fit_md."),
    json_out: bool = typer.Option(False, "--json", help="Print summary JSON to stdout only."),
    soft_fail: bool = typer.Option(False, "--soft-fail", help="Exit 0 even if some items fail."),
) -> None:
    emit_set = _parse_emit(emit)
    try:
        urls = load_manifest(path_or_dash)
    except Exception as exc:
        if not json_out:
            typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2)
    try:
        summary, exit_code = run_consumer(
            urls,
            command="get-manifest",
            out_dir=out,
            emit=emit_set,
            soft_fail=soft_fail,
        )
    except Exception as exc:
        if not json_out:
            typer.echo(f"fatal: {exc}", err=True)
        raise typer.Exit(code=3)
    if json_out:
        sys.stdout.write(json.dumps(summary, ensure_ascii=False) + "\n")
    raise typer.Exit(code=exit_code)

