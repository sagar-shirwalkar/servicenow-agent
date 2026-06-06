"""MLX backend for embedding.

Apple's MLX framework, hand-rolled BGE-base-en-v1.5 (vanilla BERT
architecture, 12 layers, 768 hidden, 12 heads, ~110M params).
Provides true ANE/GPU acceleration on M-series without going
through the unstable ONNX->CoreML bridge.

Weight conversion is a one-time cost: use
``tools/convert_bge_to_mlx.py`` to produce
``~/.cache/atlas/models/bge-base-en-v1.5-mlx/`` from the original
Hugging Face PyTorch checkpoint. The conversion only requires
``torch``; once converted, MLX inference has zero PyTorch in the
runtime path.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from transformers import AutoTokenizer

if TYPE_CHECKING:
    import mlx.core as mx

from .base import (
    EMBEDDING_DIM,
    MAX_SEQ_LENGTH,
    Embedder,
    l2_normalize,
)


def _import_mlx():
    """Import mlx.core and mlx.nn, failing loudly if not present."""
    try:
        import mlx.core as mx
        import mlx.nn as nn
    except ImportError as e:
        raise RuntimeError(
            "MLX is not installed. Run `uv sync --extra mlx` on Apple "
            "Silicon, or set ATLAS_EMBED_BACKEND=onnx-cpu to fall back."
        ) from e
    return mx, nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class _SelfAttention:
    """Multi-head self-attention block (BERT-style, no causal mask).

    Returns the projected context (output projection included). Does
    NOT include the residual add or post-LayerNorm; those happen in
    the parent ``_BertLayer`` to match BERT's BertSelfOutput
    semantics.
    """

    def __init__(self, dim: int, n_heads: int):
        mx, nn = _import_mlx()
        self.mx = mx
        self.nn = nn
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x, mask):
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        # mask has shape (B, 1, 1, L) additive: 0 for real, -1e9 for pad
        scores = scores + mask
        weights = self.mx.softmax(scores, axis=-1)
        out = (weights @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class _BertLayer:
    """One BERT transformer block: post-norm attention + post-norm FFN.

    Matches HuggingFace BERT's BertLayer.forward exactly:

        self_output    = self.self(hidden_states, attention_mask)  # Q,K,V,attn,concat
        attn_output    = self.output(self_output, hidden_states)   # dense + LN(.) + residual
        intermediate_o = self.intermediate(attn_output)            # dense + GELU
        layer_output   = self.output(intermediate_o, attn_output)  # dense + LN(.) + residual

    BERT uses post-norm throughout (LayerNorm AFTER the residual
    add, not before the sublayer). We follow the same convention.
    The dense projections are folded into the SelfAttention block
    here (mathematically equivalent) so the call structure is
    self-attention -> output projection -> residual+LN -> FFN up
    -> GELU -> FFN down -> residual+LN.
    """

    def __init__(self, dim: int, n_heads: int, ff_dim: int):
        _mx_nn = _import_mlx()
        nn = _mx_nn[1]
        self.attention = _SelfAttention(dim, n_heads)
        self.attention_output_layernorm = nn.LayerNorm(dim)
        self.intermediate = nn.Linear(dim, ff_dim)    # FFN up
        self.output = nn.Linear(ff_dim, dim)          # FFN down
        self.ffn_output_layernorm = nn.LayerNorm(dim)

    def __call__(self, x, mask):
        attn = self.attention(x, mask)
        # BertSelfOutput: LN(dense(attn) + x). dense is inside self.attention.
        attn_output = self.attention_output_layernorm(attn + x)
        # BertIntermediate: GELU(dense(attn_output))
        ff_inter = _gelu(self.intermediate(attn_output))
        # BertOutput: LN(dense(ff_inter) + attn_output)
        layer_output = self.ffn_output_layernorm(self.output(ff_inter) + attn_output)
        return layer_output


def _gelu(x):
    """GELU activation, exact match to BERT's `nn.GELU()` (no tanh approx)."""
    mx, _ = _import_mlx()
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


class BgeModel:
    """Hand-rolled BERT model matching Hugging Face BGE-base-en-v1.5.

    Architecture:
      - vocab_size=30522, hidden=768, 12 layers, 12 heads, ff=3072
      - Learned absolute position embeddings (max 512)
      - Learned token-type embeddings (size 2)
      - LayerNorm at input (BERT-style, post-norm elsewhere)
      - No final encoder LayerNorm (BGE drops it for retrieval)
      - No causal mask (BGE is bidirectional, used for retrieval)
    """

    def __init__(self, vocab_size: int = 30522, dim: int = 768,
                 n_layers: int = 12, n_heads: int = 12, ff_dim: int = 3072,
                 max_seq: int = 512):
        _mx, nn = _import_mlx()
        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.position_embeddings = nn.Embedding(max_seq, dim)
        self.token_type_embeddings = nn.Embedding(2, dim)
        self.embeddings_ln = nn.LayerNorm(dim)
        self.encoder_layers = [_BertLayer(dim, n_heads, ff_dim) for _ in range(n_layers)]

    def __call__(self, input_ids, attention_mask, token_type_ids=None):
        mx, _ = _import_mlx()
        B, L = input_ids.shape
        positions = mx.broadcast_to(mx.arange(L)[None, :], (B, L))
        h = self.word_embeddings(input_ids)
        h = h + self.position_embeddings(positions)
        if token_type_ids is None:
            token_type_ids = mx.zeros((B, L), dtype=mx.int32)
        h = h + self.token_type_embeddings(token_type_ids)
        h = self.embeddings_ln(h)
        # additive attention mask: 0 for real, -1e9 for pad
        pad = (1.0 - attention_mask.astype(mx.float32)) * -1e9
        mask = pad[:, None, None, :]
        for layer in self.encoder_layers:
            h = layer(h, mask)
        return h


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------


def _load_mlx_weights(model: BgeModel, weights_path: Path) -> None:
    """Load MLX-format weights from a directory of .npy files.

    The conversion script ``tools/convert_bge_to_mlx.py`` produces
    a directory with one ``.npy`` per weight tensor, named with
    MLX-flavored keys. We assign them onto the live model.
    """
    mx, _ = _import_mlx()

    def _npy(name: str) -> mx.array:
        arr = np.load(weights_path / f"{name}.npy")
        return mx.array(arr)

    # Embeddings
    model.word_embeddings.weight = _npy("word_embeddings.weight")
    model.position_embeddings.weight = _npy("position_embeddings.weight")
    model.token_type_embeddings.weight = _npy("token_type_embeddings.weight")
    model.embeddings_ln.weight = _npy("embeddings.LayerNorm.weight")
    model.embeddings_ln.bias = _npy("embeddings.LayerNorm.bias")

    # Encoder layers
    for i, layer in enumerate(model.encoder_layers):
        attn = layer.attention
        attn.q_proj.weight = _npy(f"encoder.layer.{i}.attention.q_proj.weight")
        attn.q_proj.bias = _npy(f"encoder.layer.{i}.attention.q_proj.bias")
        attn.k_proj.weight = _npy(f"encoder.layer.{i}.attention.k_proj.weight")
        attn.k_proj.bias = _npy(f"encoder.layer.{i}.attention.k_proj.bias")
        attn.v_proj.weight = _npy(f"encoder.layer.{i}.attention.v_proj.weight")
        attn.v_proj.bias = _npy(f"encoder.layer.{i}.attention.v_proj.bias")
        attn.o_proj.weight = _npy(f"encoder.layer.{i}.attention.o_proj.weight")
        attn.o_proj.bias = _npy(f"encoder.layer.{i}.attention.o_proj.bias")
        # Post-LN after attention (BertSelfOutput.LayerNorm)
        layer.attention_output_layernorm.weight = _npy(f"encoder.layer.{i}.attention.LayerNorm.weight")
        layer.attention_output_layernorm.bias = _npy(f"encoder.layer.{i}.attention.LayerNorm.bias")
        # FFN up
        layer.intermediate.weight = _npy(f"encoder.layer.{i}.ffn.lin1.weight")
        layer.intermediate.bias = _npy(f"encoder.layer.{i}.ffn.lin1.bias")
        # FFN down
        layer.output.weight = _npy(f"encoder.layer.{i}.ffn.lin2.weight")
        layer.output.bias = _npy(f"encoder.layer.{i}.ffn.lin2.bias")
        # Post-LN after FFN (BertOutput.LayerNorm)
        layer.ffn_output_layernorm.weight = _npy(f"encoder.layer.{i}.ffn.LayerNorm.weight")
        layer.ffn_output_layernorm.bias = _npy(f"encoder.layer.{i}.ffn.LayerNorm.bias")


# ---------------------------------------------------------------------------
# Public embedder
# ---------------------------------------------------------------------------


DEFAULT_MLX_CACHE = Path.home() / ".cache" / "atlas" / "models" / "bge-base-en-v1.5-mlx"


class MlxEmbedder(Embedder):
    """MLX-based embedder. Apple Silicon only. ANE/GPU accelerated."""

    backend = "mlx"

    def __init__(self, model_dir: str | Path) -> None:
        mx, _ = _import_mlx()

        # Resolve model_dir: either a local dir with .npy weights, or
        # the HF id (in which case we look in DEFAULT_MLX_CACHE).
        p = Path(model_dir)
        if p.is_dir() and (p / "word_embeddings.weight.npy").is_file():
            weights_path = p
        elif DEFAULT_MLX_CACHE.is_dir() and (DEFAULT_MLX_CACHE / "word_embeddings.weight.npy").is_file():
            weights_path = DEFAULT_MLX_CACHE
        else:
            raise FileNotFoundError(
                f"MLX weights not found. Looked in {p} and {DEFAULT_MLX_CACHE}. "
                f"Run `python tools/convert_bge_to_mlx.py` to convert the "
                f"Hugging Face checkpoint first."
            )

        self.model_id = "Xenova/bge-base-en-v1.5"
        self.resolved_dir = weights_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = BgeModel()
        _load_mlx_weights(self.model, weights_path)
        # Force a small eval to warm the ANE/GPU
        self._warmup_done = False
        self.active_provider = "MLX/ANE+GPU"

    def _ensure_warmup(self) -> None:
        if self._warmup_done:
            return
        mx, _ = _import_mlx()
        # one tiny forward pass to compile the graph
        _ = self._forward(["warmup"])
        mx.eval(_)
        self._warmup_done = True

    def _forward(self, texts: list[str]) -> mx.array:
        mx, _ = _import_mlx()
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )
        input_ids = mx.array(encoded["input_ids"].astype(np.int32))
        attention_mask = mx.array(encoded["attention_mask"].astype(np.int32))
        token_type_ids = mx.array(encoded.get("token_type_ids", np.zeros_like(encoded["input_ids"])).astype(np.int32))
        return self.model(input_ids, attention_mask, token_type_ids)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        mx, _ = _import_mlx()
        self._ensure_warmup()
        hidden = self._forward(texts)
        mx.eval(hidden)
        h = np.array(hidden)
        mask = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )["attention_mask"].astype(np.float32)
        # use the shared mean_pool from base
        from .base import mean_pool
        pooled = mean_pool(h, mask)
        return l2_normalize(pooled)
