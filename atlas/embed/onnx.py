"""ONNX Runtime backend for embedding.

Portable, works on every platform with ``onnxruntime`` installed.
On Apple Silicon, the CoreML execution provider is intentionally
NOT used here: the BGE model is unstable under CoreML for long
sequences (the second inference batch reliably triggers a SIGKILL
on M-series) and is 30-40x slower than CPU when it does run. CUDA
is honoured if the user explicitly asks for ``onnx-gpu``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from transformers.utils import cached_file

from .base import (
    Embedder,
    l2_normalize,
    mean_pool,
)


def _resolve_model_dir(model_dir: str | Path) -> Path:
    """Resolve a model id or local path to a local directory.

    If ``model_dir`` is an existing directory, return it as-is. If
    it is a Hugging Face model id, use ``cached_file`` to trigger
    the download and resolve the path to the ONNX file's parent
    directory (``<snapshot>/onnx/``).
    """
    p = Path(model_dir)
    if p.is_dir():
        return p
    onnx = cached_file(model_dir, "onnx/model.onnx")
    return Path(onnx).parent.parent


def _providers(prefer_gpu: bool) -> list[str]:
    """Pick execution providers in order, GPU only if explicitly asked."""
    available = ort.get_available_providers()
    wanted: list[str] = []
    if prefer_gpu and "CUDAExecutionProvider" in available:
        wanted.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        wanted.append("CPUExecutionProvider")
    return wanted


class OnnxEmbedder(Embedder):
    """ONNX Runtime embedder. Portable, slow-but-stable, no special deps."""

    backend = "onnx"

    def __init__(self, model_id: str | Path, prefer_gpu: bool = False) -> None:
        self.model_id = str(model_id)
        self.resolved_dir = _resolve_model_dir(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.resolved_dir)
        providers = _providers(prefer_gpu)
        if not providers:
            raise RuntimeError("No ONNX execution providers available")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        onnx_path = cached_file(self.model_id, "onnx/model.onnx")
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=providers,
        )
        self.active_provider = self.session.get_providers()[0]
        self._prefer_gpu = prefer_gpu

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        feed = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        if "token_type_ids" in encoded:
            feed["token_type_ids"] = encoded["token_type_ids"]
        outputs = self.session.run(None, feed)
        pooled = mean_pool(outputs[0], encoded["attention_mask"])
        return l2_normalize(pooled)
