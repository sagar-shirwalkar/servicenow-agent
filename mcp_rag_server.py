"""MCP server exposing the pre-built ServiceNow RAG bundle.

Reads the portable bundle produced by ``make_bundle.py`` and answers
semantic-search queries by:
  1. Embedding the query with the bundled ONNX model.
  2. Computing cosine similarity against the precomputed matrix.
  3. Returning the top-k chunks with their metadata.

The bundle is loaded once at startup. Bundle format (see
``make_bundle.py``) is platform-agnostic; at runtime on Apple
Silicon we request the CoreML execution provider for ANE
acceleration, falling back to CPU elsewhere.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from embed import Embedder, load_embeddings

app = Server("servicenow-rag")


class Bundle:
    def __init__(self, bundle_dir: Path) -> None:
        self.bundle_dir = bundle_dir.resolve()
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Bundle manifest missing: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text())
        chunks_path = self.bundle_dir / "chunks.parquet"
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Bundle chunks missing: {chunks_path}")
        self.chunks = pd.read_parquet(chunks_path)
        self.embeddings = load_embeddings(self.bundle_dir)
        norms_path = self.bundle_dir / "norms.f32.npy"
        if norms_path.is_file():
            self.norms = np.load(norms_path)
        else:
            self.norms = np.linalg.norm(self.embeddings, axis=1).astype(np.float32)
        model_dir = self.bundle_dir / "model"
        if not (model_dir / "onnx" / "model.onnx").is_file():
            raise FileNotFoundError(f"Bundle model missing: {model_dir}/onnx/model.onnx")
        self.embedder = Embedder(model_dir)
        self._norms_safe = self.norms.clip(min=1e-9)

    def search(
        self,
        query: str,
        top_k: int = 5,
        publication: str | None = None,
        product_area: str | None = None,
        is_code: bool | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        q = self.embedder.embed([query])[0]
        scores = (self.embeddings @ q).flatten() / self._norms_safe

        mask = np.ones(len(self.chunks), dtype=bool)
        if publication:
            mask &= (self.chunks["publication"].values == publication)
        if product_area:
            mask &= (self.chunks["product_area"].values == product_area)
        if is_code is not None:
            mask &= (self.chunks["is_code"].values == bool(is_code))

        masked_scores = np.where(mask, scores, -np.inf)
        k = min(top_k, int(mask.sum()))
        if k == 0:
            return []
        top_idx = np.argpartition(-masked_scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-masked_scores[top_idx])]

        results: list[dict[str, Any]] = []
        for i in top_idx:
            score = float(scores[i])
            if score < min_score:
                continue
            row = self.chunks.iloc[i]
            results.append(
                {
                    "id": row["id"],
                    "score": score,
                    "publication": row["publication"],
                    "file": row["file"],
                    "heading": row["heading"],
                    "title": row["title"],
                    "product_area": row["product_area"],
                    "last_updated": row["last_updated"],
                    "canonical_url": row["canonical_url"],
                    "is_code": bool(row["is_code"]),
                    "text": row["text"],
                }
            )
        return results

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        match = self.chunks[self.chunks["id"] == chunk_id]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "id": row["id"],
            "publication": row["publication"],
            "file": row["file"],
            "heading": row["heading"],
            "title": row["title"],
            "product_area": row["product_area"],
            "last_updated": row["last_updated"],
            "canonical_url": row["canonical_url"],
            "is_code": bool(row["is_code"]),
            "text": row["text"],
        }


def _result(payload: Any) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]


@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_docs",
            description="Semantic search across ServiceNow documentation. Best for conceptual or 'how do I' queries. Returns top-k chunks with file paths, headings, and similarity scores.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                    "publication": {"type": "string"},
                    "product_area": {"type": "string"},
                    "min_score": {"type": "number", "default": 0.0, "minimum": -1.0, "maximum": 1.0},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="search_code",
            description="Semantic search restricted to code examples in the docs. Useful for 'show me a script that does X'.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                    "publication": {"type": "string"},
                    "min_score": {"type": "number", "default": 0.0},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_chunk",
            description="Fetch a single chunk by its ID. Use after a search_docs call to get the full content of a specific hit.",
            inputSchema={
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
        ),
        Tool(
            name="get_bundle_info",
            description="Return the bundle manifest: source repo/branch/SHA, build date, chunk count, embedding model. Use to cite freshness.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        bundle = _bundle_cache(ARGS.bundle)
    except FileNotFoundError as e:
        return _result({"error": str(e)})

    if name == "get_bundle_info":
        return _result(bundle.manifest)

    if name == "search_docs":
        return _result(
            bundle.search(
                arguments["query"],
                top_k=arguments.get("top_k", 5),
                publication=arguments.get("publication"),
                product_area=arguments.get("product_area"),
                min_score=arguments.get("min_score", 0.0),
            )
        )

    if name == "search_code":
        return _result(
            bundle.search(
                arguments["query"],
                top_k=arguments.get("top_k", 5),
                publication=arguments.get("publication"),
                is_code=True,
                min_score=arguments.get("min_score", 0.0),
            )
        )

    if name == "get_chunk":
        chunk = bundle.get_chunk(arguments["chunk_id"])
        if chunk is None:
            return _result({"error": f"chunk_id not found: {arguments['chunk_id']}"})
        return _result(chunk)

    raise ValueError(f"Unknown tool: {name}")


def _bundle_cache(bundle_arg: str) -> Bundle:
    if not hasattr(_bundle_cache, "_instance"):
        bundle_path = Path(bundle_arg).expanduser()
        if not bundle_path.is_absolute():
            bundle_path = bundle_path.resolve()
        _bundle_cache._instance = Bundle(bundle_path)
    return _bundle_cache._instance


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ServiceNow RAG MCP server")
    p.add_argument("--bundle", required=True, help="Path to a pre-built RAG bundle directory")
    return p.parse_args()


ARGS = parse_args()


async def _main() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_main())
