"""H2-boundary markdown chunking with YAML frontmatter parsing.

Markdown → chunks. Splits on `## ` headers, parses YAML frontmatter,
flags code-containing chunks, falls back to paragraph splitting for
oversized sections. No AST, no regex horrors.

The ServiceNowDocs repo authors already chose their chunk size: one
``.md`` file per topic, with ``##`` (H2) sections as natural
sub-chunks. This module respects that structure instead of imposing a
sliding-window splitter that would cut across the docs team's
deliberate boundaries.

Each output chunk carries:
  * heading     - the H2 text immediately preceding the chunk
  * text        - the chunk body, stripped of leading/trailing blanks
  * frontmatter - the parsed YAML frontmatter (title, product_area,
                  last_updated, canonical_url)
  * publication - the parent folder name (e.g. ``it-service-management``)
  * file        - the source file relative to ``markdown/``
  * is_code     - True if the chunk is dominated by a fenced code block

Chunks are emitted in document order. Empty chunks (e.g. an H2
header with no body) are dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n(.*)\Z", re.DOTALL)
_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_MAX_CHUNK_CHARS = 8000


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from a markdown body.

    Returns ``(metadata, body)``. If no frontmatter is present,
    ``metadata`` is an empty dict and ``body`` is the input unchanged.
    Malformed frontmatter is logged (via exception swallow) and treated
    as no frontmatter; we never want a single bad file to fail the
    whole bundle build.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text
    raw_meta, body = match.group(1), match.group(2)
    try:
        meta = yaml.safe_load(raw_meta) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, body


def _split_on_h2(body: str) -> list[tuple[str, str]]:
    """Split body on H2 boundaries.

    Returns a list of ``(heading, section_text)`` pairs. The first
    pair uses ``"Overview"`` as the heading if the body starts with
    prose before any H2.
    """
    matches = list(_H2_RE.finditer(body))
    if not matches:
        return [("Overview", body.strip())]

    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        prologue = body[: matches[0].start()].strip()
        if prologue:
            sections.append(("Overview", prologue))

    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        section_text = body[start:end].strip()
        if section_text:
            sections.append((heading, section_text))

    return sections


def _is_code_chunk(text: str) -> bool:
    """Heuristic: chunk 'contains code' if it has any fenced block.

    We only need a cheap signal for the ``is_code`` flag used by
    ``mcp_rag_server.search_code``. The check is line-based so it
    is robust against prose interleaved with code - which is the
    common pattern in ServiceNow docs.
    """
    return _FENCE_RE.search(text) is not None


def _hard_split(text: str, max_chars: int) -> list[str]:
    """Last-resort split for chunks that exceed the embedding limit.

    Tries to break on a paragraph blank line; falls back to a hard
    cut. Should rarely trigger because the docs team keeps H2
    sections reasonably sized.
    """
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    while text:
        if len(text) <= max_chars:
            parts.append(text)
            break
        cut = text.rfind("\n\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = text.rfind("\n", 0, max_chars)
        if cut < max_chars // 2:
            cut = max_chars
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    return [p for p in parts if p]


def chunk_markdown(
    text: str,
    publication: str,
    file: str,
) -> list[dict]:
    """Chunk a single markdown file into H2-boundary records.

    ``publication`` and ``file`` are stamped into every chunk as
    metadata. The ``file`` argument should be the path relative to
    the ``markdown/`` root (e.g. ``it-service-management/incident.md``).
    """
    frontmatter, body = parse_frontmatter(text)
    sections = _split_on_h2(body)

    chunks: list[dict] = []
    for heading, section_text in sections:
        for piece in _hard_split(section_text, _MAX_CHUNK_CHARS):
            chunks.append(
                {
                    "heading": heading,
                    "text": piece,
                    "frontmatter": frontmatter,
                    "publication": publication,
                    "file": file,
                    "is_code": _is_code_chunk(piece),
                }
            )
    return chunks


def chunk_file(path: Path, repo_root: Path) -> list[dict]:
    """Read a markdown file from disk and chunk it.

    ``path`` is the absolute file path; ``repo_root`` is the repo
    root. The publication name is the first directory under
    ``markdown/``; the file field is the relative path from
    ``markdown/``.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    rel = path.relative_to(repo_root)
    parts = rel.parts
    if len(parts) < 2 or parts[0] != "markdown":
        return []
    publication = parts[1]
    file = "/".join(parts[2:]) if len(parts) > 2 else parts[1]
    return chunk_markdown(text, publication, file)
