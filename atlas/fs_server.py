"""MCP server exposing the ServiceNowDocs repo as a navigable filesystem.

Five tools, no models, no embeddings. Backed by a local ``git clone``
of the docs repo plus ripgrep for full-text search. This is the
"give the model the link" surface - deterministic, zero-infra,
reproducible across every client that speaks MCP.

The same server works for any markdown repo - swap the path and
it'll happily expose a different docs site.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

app = Server("servicenow-fs")


def _repo_root(args_repo: str) -> Path:
    p = Path(args_repo).expanduser().resolve()
    if not (p / "markdown").is_dir():
        raise FileNotFoundError(
            f"No 'markdown/' directory at {p}. Did you clone ServiceNowDocs?"
        )
    return p


def _parse_md(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    meta: dict[str, Any] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            try:
                loaded = yaml.safe_load(parts[1]) or {}
                if isinstance(loaded, dict):
                    meta = loaded
            except yaml.YAMLError:
                pass
            body = parts[2].strip()
    return {"frontmatter": meta, "content": body}


def list_publications(root: Path) -> list[dict[str, Any]]:
    md = root / "markdown"
    out: list[dict[str, Any]] = []
    for pub_dir in sorted(p for p in md.iterdir() if p.is_dir()):
        files = list(pub_dir.rglob("*.md"))
        out.append(
            {
                "name": pub_dir.name,
                "file_count": len(files),
            }
        )
    return out


def list_publication_files(root: Path, publication: str) -> list[dict[str, Any]]:
    pub_dir = root / "markdown" / publication
    if not pub_dir.is_dir():
        raise FileNotFoundError(f"Publication not found: {publication}")
    out: list[dict[str, Any]] = []
    for f in sorted(pub_dir.rglob("*.md")):
        parsed = _parse_md(f)
        meta = parsed["frontmatter"]
        out.append(
            {
                "file": str(f.relative_to(pub_dir)),
                "title": meta.get("title", ""),
                "product_area": meta.get("product_area", ""),
                "last_updated": str(meta.get("last_updated", "")),
            }
        )
    return out


def read_publication_file(
    root: Path, publication: str, file: str, max_chars: int = 50_000
) -> dict[str, Any]:
    target = (root / "markdown" / publication / file).resolve()
    pub_root = (root / "markdown" / publication).resolve()
    if not str(target).startswith(str(pub_root)):
        raise ValueError("Path traversal blocked")
    if not target.is_file():
        raise FileNotFoundError(f"{publication}/{file} not found")
    parsed = _parse_md(target)
    content = parsed["content"]
    truncated = False
    if len(content) > max_chars:
        content = content[:max_chars]
        truncated = True
    return {
        "publication": publication,
        "file": file,
        "frontmatter": parsed["frontmatter"],
        "content": content,
        "truncated": truncated,
    }


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def get_release_info(root: Path) -> dict[str, Any]:
    sha = _git("rev-parse", "HEAD", cwd=root)
    short = _git("rev-parse", "--short", "HEAD", cwd=root)
    branch = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)
    last_commit_date = _git("log", "-1", "--format=%cI", cwd=root)
    return {
        "branch": branch,
        "sha": sha,
        "short_sha": short,
        "last_commit_date": last_commit_date,
        "file_count": sum(1 for _ in root.rglob("*.md")),
    }


def full_text_search(
    root: Path,
    query: str,
    scope: str | None = None,
    regex: bool = False,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    if not shutil.which("rg"):
        raise RuntimeError("ripgrep (rg) not installed. Install via `brew install ripgrep`.")
    search_root = root / "markdown" / scope if scope else root / "markdown"
    if not search_root.is_dir():
        raise FileNotFoundError(f"Scope not found: {scope}")
    cmd = ["rg", "--no-heading", "--line-number", "--color", "never"]
    if not regex:
        cmd.append("--fixed-strings")
    cmd.extend(["--max-count", "1", query, str(search_root)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    hits: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines()[:max_results]:
        if ":" not in line:
            continue
        path_part, _, rest = line.partition(":")
        line_no_str, _, _ = rest.partition(":")
        try:
            line_no = int(line_no_str)
        except ValueError:
            continue
        file_path = Path(path_part)
        try:
            rel = file_path.relative_to(root / "markdown")
        except ValueError:
            rel = file_path
        hits.append(
            {
                "file": str(rel),
                "line": line_no,
                "preview": _preview_line(file_path, line_no),
            }
        )
    return hits


def _preview_line(path: Path, line_no: int, context: int = 1) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = max(0, line_no - 1 - context)
        end = min(len(lines), line_no + context)
        return "".join(lines[start:end]).strip()
    except OSError:
        return ""


def _result(payload: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="list_publications",
            description="List every ServiceNow documentation publication (top-level folder) available in the local clone.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="list_files",
            description="List all files in a single publication, with title/product_area/last_updated from YAML frontmatter.",
            inputSchema={
                "type": "object",
                "properties": {
                    "publication": {
                        "type": "string",
                        "description": "Publication folder name, e.g. 'it-service-management'",
                    }
                },
                "required": ["publication"],
            },
        ),
        Tool(
            name="read_file",
            description="Read a single markdown file. Returns parsed YAML frontmatter and the body content.",
            inputSchema={
                "type": "object",
                "properties": {
                    "publication": {"type": "string"},
                    "file": {
                        "type": "string",
                        "description": "Path relative to the publication folder, e.g. 'incident-management.md'",
                    },
                    "max_chars": {
                        "type": "integer",
                        "default": 50000,
                        "description": "Truncate the body to this many characters.",
                    },
                },
                "required": ["publication", "file"],
            },
        ),
        Tool(
            name="search",
            description="Full-text search (ripgrep) over the docs. Fast, deterministic, regex-capable.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "scope": {
                        "type": "string",
                        "description": "Optional: restrict to one publication folder.",
                    },
                    "regex": {"type": "boolean", "default": False},
                    "max_results": {"type": "integer", "default": 50},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_release_info",
            description="Return the current docs release metadata: branch, commit SHA, last commit date, total file count.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    root = _repo_root(ARGS.repo)
    try:
        if name == "list_publications":
            return _result(list_publications(root))
        if name == "list_files":
            return _result(list_publication_files(root, arguments["publication"]))
        if name == "read_file":
            return _result(
                read_publication_file(
                    root,
                    arguments["publication"],
                    arguments["file"],
                    arguments.get("max_chars", 50_000),
                )
            )
        if name == "search":
            return _result(
                full_text_search(
                    root,
                    arguments["query"],
                    arguments.get("scope"),
                    arguments.get("regex", False),
                    arguments.get("max_results", 50),
                )
            )
        if name == "get_release_info":
            return _result(get_release_info(root))
    except (FileNotFoundError, ValueError) as e:
        return _result({"error": str(e)})
    raise ValueError(f"Unknown tool: {name}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ServiceNow Atlas filesystem MCP server")
    p.add_argument(
        "--repo",
        default="./data/servicenow-docs/ServiceNowDocs-australia",
        help="Path to ServiceNowDocs clone",
    )
    return p.parse_args()


ARGS = parse_args()


async def serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    asyncio.run(serve())


if __name__ == "__main__":
    main()
