"""Embedding backends for the ServiceNow Atlas bundle.

Chunks → vectors. A small package with one ABC (`Embedder`) and two
backends: `OnnxEmbedder` (portable, ONNX+CPU floor, supports CUDA via
`--prefer nvidia` if `onnxruntime-gpu` is installed) and `MlxEmbedder`
(Apple Silicon, hand-rolled BGE loaded from
`~/.cache/atlas/models/bge-base-en-v1.5-mlx/`). `get_embedder()` is
the factory; `resolve_backend()` is the probe used by `atlas-doctor`.
Both backends share `mean_pool`, `l2_normalize`, `load_embeddings`,
and a 3-attempt exponential-backoff retry wrapper.

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
