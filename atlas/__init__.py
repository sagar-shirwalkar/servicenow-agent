"""ServiceNow Atlas — local-first AI knowledge layer for ServiceNow development.

Atlas is a Python package that bundles the two MCP servers, the bundle
build pipeline, and operational helpers. The name is a nod to the
Greek titan who carried the world on his shoulders: this package
carries the ServiceNow documentation and hands it to any AI agent
that speaks MCP.

Submodules
----------
chunk
    H2-boundary markdown chunker with YAML frontmatter parsing.
embed
    Layered embedding backends (ONNX+CPU floor, optional Apple MLX
    and NVIDIA CUDA fast paths). Use ``atlas.embed.get_embedder()``
    to get the best available embedder for the current host.
fs_server
    MCP server exposing the docs as a navigable filesystem.
rag_server
    MCP server exposing the pre-built vector bundle.
make_bundle
    Build a portable RAG bundle from a local ServiceNowDocs clone.
download / backup / restore
    Operational helpers for the bundle lifecycle.
doctor
    Diagnose the installation, probe all backends, validate a bundle.
smoke_test
    End-to-end validation of the build + search pipeline.

Future
------
agent
    Reasoning agent that uses the MCP servers (planned).
training
    LoRA / QLoRA fine-tuning pipeline (planned).
"""

from __future__ import annotations

__version__ = "0.3.0"
__all__ = ["__version__"]
