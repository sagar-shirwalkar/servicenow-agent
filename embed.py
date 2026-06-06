"""ONNX-based embedding with batching, progress, and retries.

Uses the pre-exported ``Xenova/bge-base-en-v1.5`` ONNX model so we
avoid pulling in ``torch`` or ``optimum`` at build time. The model
is a 110M-parameter encoder producing 768-dimensional embeddings
suitable for cosine similarity retrieval.

At build time we embed ~250k chunks in batches of 32. At runtime
(in ``mcp_rag_server.py``) we embed a single query.

The Apple Silicon runtime path enables the CoreML execution
provider, which dispatches the model to the Apple Neural Engine
when possible. Falls back to CPU on other platforms.
"""

from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

DEFAULT_MODEL_ID = "Xenova/bge-base-en-v1.5"
EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 512

_BATCH_SIZE = 32
_MAX_RETRIES = 3
_RETRY_BASE_SECONDS = 2.0


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def available_providers() -> list[str]:
    """Return the ONNX execution providers we want to try, in order.

    CoreML is requested on Apple Silicon; CPU is always the last
    fallback. CUDA is intentionally absent: this project targets
    Apple Silicon only and we don't want a silent CUDA dependency
    sneaking in via the runtime.
    """
    available = ort.get_available_providers()
    wanted: list[str] = []
    if is_apple_silicon() and "CoreMLExecutionProvider" in available:
        wanted.append("CoreMLExecutionProvider")
    if "CPUExecutionProvider" in available:
        wanted.append("CPUExecutionProvider")
    return wanted


def _mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token embeddings, respecting the attention mask.

    Standard BGE pooling. The attention mask is broadcast across the
    hidden dimension so padded positions contribute zero.
    """
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
    return (x / norms).astype(np.float32)


class Embedder:
    """Wrapper around an ONNX embedding session + tokenizer.

    ``model_dir`` can be either a local path (for the bundle's
    ``model/`` directory) or a Hugging Face model id (for first-time
    download during the build).
    """

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = str(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        providers = available_providers()
        if not providers:
            raise RuntimeError("No ONNX execution providers available")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(Path(self.model_dir) / "onnx" / "model.onnx"),
            sess_options=so,
            providers=providers,
        )
        self.active_provider = self.session.get_providers()[0]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape ``(n, 768)`` float32."""
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )
        outputs = self.session.run(
            None,
            {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
            },
        )
        pooled = _mean_pool(outputs[0], encoded["attention_mask"])
        return _l2_normalize(pooled)

    def embed_with_progress(
        self,
        texts: list[str],
        batch_size: int = _BATCH_SIZE,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Embed ``texts`` in batches with progress and retries.

        Returns shape ``(len(texts), 768)`` float32. Failed chunks
        after all retries fall back to zero vectors, which will rank
        last in cosine similarity without breaking the search.
        """
        n = len(texts)
        out = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        if n == 0:
            return out

        start = time.time()
        indexed_batches: Iterable[tuple[int, int]] = (
            (i, min(i + batch_size, n)) for i in range(0, n, batch_size)
        )
        if show_progress:
            try:
                from tqdm import tqdm

                indexed_batches = tqdm(
                    list(indexed_batches),
                    desc=f"Embedding ({self.active_provider})",
                    unit="batch",
                )
            except ImportError:
                show_progress = False

        for i, j in indexed_batches:
            batch = texts[i:j]
            attempt = 0
            while True:
                try:
                    out[i:j] = self.embed(batch)
                    break
                except Exception as e:
                    attempt += 1
                    if attempt > _MAX_RETRIES:
                        print(
                            f"\n  [embed] giving up on batch {i}-{j} after "
                            f"{_MAX_RETRIES} retries: {e}"
                        )
                        break
                    wait = _RETRY_BASE_SECONDS ** attempt
                    print(
                        f"\n  [embed] batch {i}-{j} failed (attempt {attempt}): "
                        f"{e}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)

            if show_progress and isinstance(indexed_batches, list):
                done = j
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(
                    f"  {done}/{n} chunks ({rate:.1f}/s) | ETA: {eta / 60:.1f} min",
                    end="\r",
                )

        if show_progress:
            elapsed = time.time() - start
            print(f"\n  Done: {n} chunks in {elapsed / 60:.1f} min")
        return out


def save_bundle_artifacts(
    embeddings: np.ndarray,
    output_dir: Path,
    dtype: str = "float16",
) -> tuple[Path, Path]:
    """Write embeddings and precomputed norms to disk.

    Float16 halves the on-disk size (~360MB -> ~180MB for 250k
    chunks) with negligible effect on retrieval quality because
    cosine similarity is rank-preserving under half precision.
    Norms are stored separately in float32 for accurate
    cosine-at-query-time.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if dtype == "float16":
        emb_path = output_dir / "embeddings.f16.npy"
        emb_to_save = embeddings.astype(np.float16)
    elif dtype == "float32":
        emb_path = output_dir / "embeddings.f32.npy"
        emb_to_save = embeddings.astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    np.save(emb_path, emb_to_save)
    norms = np.linalg.norm(embeddings, axis=1).astype(np.float32)
    norms_path = output_dir / "norms.f32.npy"
    np.save(norms_path, norms)
    return emb_path, norms_path


def load_embeddings(bundle_dir: Path) -> np.ndarray:
    """Load embeddings from a bundle, returning float32."""
    f16 = bundle_dir / "embeddings.f16.npy"
    f32 = bundle_dir / "embeddings.f32.npy"
    if f16.exists():
        return np.load(f16).astype(np.float32)
    if f32.exists():
        return np.load(f32).astype(np.float32)
    raise FileNotFoundError(f"No embeddings found in {bundle_dir}")
