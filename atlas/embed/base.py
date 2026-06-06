"""Abstract base class and factory for embedding backends.

The :class:`Embedder` ABC defines the public interface used by
``make_bundle.py`` and ``rag_server.py``: ``embed(texts)`` returns a
``(n, 768)`` float32 numpy array of L2-normalized vectors, and
``embed_with_progress`` adds batching + tqdm progress + retries.

Concrete implementations live in sibling modules:

- :mod:`.onnx` — ONNX Runtime + CPU. Portable, always available.
- :mod:`.mlx` — Apple's MLX framework. M-series only, ~5-10x
  faster than CPU when available.

Use :func:`get_embedder` to pick the best available backend for
the current platform, honouring any user override.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np

EMBEDDING_DIM = 768
MAX_SEQ_LENGTH = 512

# Default model used for new bundles. The ONNX export is from
# Xenova (used by the transformers.js project) and is what we ship
# the ONNX+CPU path with. The MLX conversion script reads weights
# from the matching PyTorch model ``BAAI/bge-base-en-v1.5``.
DEFAULT_MODEL_ID = "Xenova/bge-base-en-v1.5"

Backend = Literal["mlx", "onnx-cpu", "onnx-gpu", "auto"]

# Friendly CLI names -> canonical backend names.
_CANONICAL_CHOICE = {
    "apple": "mlx",
    "mlx": "mlx",
    "nvidia": "onnx-gpu",
    "gpu": "onnx-gpu",
    "cuda": "onnx-gpu",
    "onnx-gpu": "onnx-gpu",
    "cpu": "onnx-cpu",
    "onnx-cpu": "onnx-cpu",
}


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _read_config_backend() -> str | None:
    """Read ``prefer`` from ``~/.config/atlas.toml`` if present.

    No tomli dependency: we use a small line-based parser that
    handles ``[section]`` and ``key = "value"`` only, which is all
    the config file needs. A real TOML file with nested tables or
    arrays would not be supported; we document that in the README.
    """
    cfg_path = Path.home() / ".config" / "atlas.toml"
    if not cfg_path.is_file():
        return None
    try:
        in_backend = False
        for line in cfg_path.read_text().splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if s.startswith("[") and s.endswith("]"):
                in_backend = s == "[backend]"
                continue
            if in_backend and "=" in s:
                key, _, val = s.partition("=")
                if key.strip() == "prefer":
                    return val.strip().strip('"').strip("'")
    except OSError:
        return None
    return None


def has_nvidia_gpu() -> bool:
    """Return True if a CUDA-capable GPU is visible to the system."""
    if not shutil.which("nvidia-smi"):
        return False
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.returncode == 0 and "GPU" in out.stdout
    except (subprocess.TimeoutExpired, OSError):
        return False


def has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401
        return True
    except ImportError:
        return False


def has_onnxruntime_gpu() -> bool:
    try:
        import onnxruntime as ort  # noqa: F401
        return "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        return False


def resolve_backend(prefer: str | None = None) -> tuple[str, str]:
    """Pick the best backend for this machine.

    Returns ``(backend, reason)`` where ``backend`` is one of
    ``"mlx"``, ``"onnx-gpu"``, ``"onnx-cpu"`` and ``reason`` is a
    short human-readable string explaining the choice.

    Resolution order (highest priority first):
    1. ``prefer`` argument (if not ``"auto"`` / ``None``) is honoured
       if its required deps are importable; else we fall through.
    2. ``ATLAS_EMBED_BACKEND`` env var (same semantics as ``prefer``).
    3. ``~/.config/atlas.toml`` ``[backend]`` ``prefer = "..."`` value.
    4. Auto-detect: MLX on Apple Silicon, onnx-gpu on NVIDIA, else
       onnx-cpu.

    The ``prefer`` argument (and the env var / config value) accept
    either the friendly CLI names (``"apple"``, ``"nvidia"``,
    ``"cpu"``) or the canonical backend names (``"mlx"``,
    ``"onnx-gpu"``, ``"onnx-cpu"``).
    """
    config_value = _read_config_backend()
    raw = prefer or os.environ.get("ATLAS_EMBED_BACKEND") or config_value or "auto"
    choice = _CANONICAL_CHOICE.get(raw.lower(), raw.lower())

    def _ok(name: str) -> bool:
        if name == "mlx":
            return has_mlx() and is_apple_silicon()
        if name == "onnx-gpu":
            return has_onnxruntime_gpu() and has_nvidia_gpu()
        if name == "onnx-cpu":
            return True
        return False

    if choice == "mlx":
        if _ok("mlx"):
            return "mlx", "user override (or auto) and MLX available on Apple Silicon"
        return "onnx-cpu", "MLX requested but not importable; using ONNX+CPU fallback"
    if choice == "onnx-gpu":
        if _ok("onnx-gpu"):
            return "onnx-gpu", "user override and CUDA provider available"
        return "onnx-cpu", "CUDA requested but no NVIDIA GPU; using ONNX+CPU fallback"
    if choice == "onnx-cpu":
        return "onnx-cpu", "user requested ONNX+CPU"

    # auto: probe in preference order
    if is_apple_silicon() and has_mlx():
        return "mlx", "Apple Silicon detected and MLX is importable"
    if has_nvidia_gpu() and has_onnxruntime_gpu():
        return "onnx-gpu", "NVIDIA GPU detected and CUDA provider is available"
    return "onnx-cpu", "no fast path available; using portable ONNX+CPU"


def get_embedder(model_id: str | Path, prefer: str | None = None) -> "Embedder":
    """Construct an :class:`Embedder` for the best available backend.

    ``model_id`` can be a Hugging Face model id (``"Xenova/bge-base-en-v1.5"``)
    or a local directory (the bundle's ``model/`` dir, or a converted
    MLX weights dir). The factory resolves the backend first, then
    delegates to the backend-specific constructor.
    """
    backend, _reason = resolve_backend(prefer)
    if backend == "mlx":
        from .mlx import MlxEmbedder
        return MlxEmbedder(model_id)
    if backend == "onnx-gpu":
        from .onnx import OnnxEmbedder
        return OnnxEmbedder(model_id, prefer_gpu=True)
    from .onnx import OnnxEmbedder
    return OnnxEmbedder(model_id, prefer_gpu=False)


class Embedder(ABC):
    """Backend-agnostic embedding interface.

    Subclasses must implement :meth:`embed`. The default
    :meth:`embed_with_progress` adds batching, tqdm progress, and
    retry logic that should work for any backend.
    """

    backend: str  # set by subclass
    active_provider: str  # set by subclass
    model_id: str  # set by subclass
    resolved_dir: Path  # set by subclass

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts. Returns shape ``(n, 768)`` float32,
        L2-normalized (unit vectors)."""

    def embed_with_progress(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        max_retries: int = 3,
    ) -> np.ndarray:
        """Embed ``texts`` in batches with progress and retries.

        Failed chunks after all retries fall back to zero vectors,
        which rank last in cosine similarity without breaking search.
        """
        import time
        n = len(texts)
        out = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        if n == 0:
            return out

        start = time.time()
        indices = list(range(0, n, batch_size))

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(indices, desc=f"Embedding ({self.active_provider})", unit="batch")
            except ImportError:
                iterator = indices
                show_progress = False
        else:
            iterator = indices

        for i in iterator:
            j = min(i + batch_size, n)
            batch = texts[i:j]
            attempt = 0
            while True:
                try:
                    out[i:j] = self.embed(batch)
                    break
                except Exception as e:  # noqa: BLE001
                    attempt += 1
                    if attempt > max_retries:
                        print(
                            f"\n  [embed] giving up on batch {i}-{j} after "
                            f"{max_retries} retries: {e}"
                        )
                        break
                    wait = 2.0 ** attempt
                    print(
                        f"\n  [embed] batch {i}-{j} failed (attempt {attempt}): "
                        f"{e}. Retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)

            if not show_progress:
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


def mean_pool(last_hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean-pool token embeddings, respecting the attention mask."""
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = mask.sum(axis=1).clip(min=1e-9)
    return summed / counts


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D array. Output is float32."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True).clip(min=1e-12)
    return (x / norms).astype(np.float32)


def load_embeddings(bundle_dir: Path) -> np.ndarray:
    """Load embeddings from a bundle, preferring float16."""
    f16 = bundle_dir / "embeddings.f16.npy"
    if f16.is_file():
        return np.load(f16).astype(np.float32)
    f32 = bundle_dir / "embeddings.f32.npy"
    if f32.is_file():
        return np.load(f32)
    raise FileNotFoundError(f"No embeddings found in {bundle_dir}")
