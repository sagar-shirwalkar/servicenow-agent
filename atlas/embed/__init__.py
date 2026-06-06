"""Embedding backends for the ServiceNow Atlas bundle.

The bundle is backend-agnostic (just float16 vectors). The runtime
inference can run on:

- :class:`.onnx.OnnxEmbedder` — ONNX Runtime + CPU. Portable, always
  available. ~4 ms per short query.
- :class:`.mlx.MlxEmbedder` — Apple MLX on M-series. ~1-2 ms per
  query, 5-10x faster at build time, no ONNX bridge.

Use :func:`get_embedder` to pick the best available backend for the
current platform. Use :func:`resolve_backend` to query which one
will be chosen (used by ``atlas-doctor``).
"""

from .base import (
    DEFAULT_MODEL_ID,
    Embedder,
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    get_embedder,
    resolve_backend,
    has_mlx,
    has_nvidia_gpu,
    has_onnxruntime_gpu,
    is_apple_silicon,
    load_embeddings,
    l2_normalize,
    mean_pool,
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "Embedder",
    "EMBEDDING_DIM",
    "MAX_SEQ_LENGTH",
    "get_embedder",
    "resolve_backend",
    "has_mlx",
    "has_nvidia_gpu",
    "has_onnxruntime_gpu",
    "is_apple_silicon",
    "load_embeddings",
    "l2_normalize",
    "mean_pool",
]
