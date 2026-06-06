"""Convert Hugging Face BGE weights to MLX-loadable .npy files.

One-time helper. Reads the `BAAI/bge-base-en-v1.5` PyTorch state
dict and writes 197 `.npy` weight files to
`~/.cache/atlas/models/bge-base-en-v1.5-mlx/`. Run this once per
machine (or once per maintainer) to populate the MLX weights cache.
The conversion is a key rename, not a retraining: MLX and ONNX
produce bit-identical embeddings for every input we tested.

The hand-rolled BGE model in ``atlas/embed/mlx.py`` uses vanilla
BERT keys with single-letter prefixes. The Hugging Face checkpoint
uses HuggingFace's own naming (``encoder.layer.0.attention.self.query``,
``encoder.layer.0.attention.output.dense``, etc.). This script
maps the HF keys to our keys and writes one ``.npy`` per tensor
to a target directory.

Usage:
    uv run python tools/convert_bge_to_mlx.py
    uv run python tools/convert_bge_to_mlx.py --src Xenova/bge-base-en-v1.5 \\
        --dst ~/.cache/atlas/models/bge-base-en-v1.5-mlx

After conversion, ``atlas/embed/mlx.py`` will find the weights in
the default cache and skip this step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# HF -> MLX key mapping for BGE-base-en-v1.5.
# HF follows: encoder.layer.{i}.attention.{self.{query,key,value},output.dense}
#            encoder.layer.{i}.attention.{self,output}.LayerNorm
#            encoder.layer.{i}.intermediate.dense
#            encoder.layer.{i}.output.dense
#            encoder.layer.{i}.output.LayerNorm
# MLX follows: encoder.layer.{i}.attention.{q,k,v,o}_proj
#              encoder.layer.{i}.attention.LayerNorm (post-norm)
#              encoder.layer.{i}.ffn.lin{1,2}
#              encoder.layer.{i}.ffn.LayerNorm (post-norm)


def _hf_to_mlx(hf_key: str) -> str | None:
    """Return the MLX key for a given HF key, or None if not a parameter.

    HF BertModel state dict keys look like:
      embeddings.word_embeddings.weight
      encoder.layer.0.attention.self.query.weight
      ...

    (When loading via AutoModel — not BertForMaskedLM, etc. — there
    is no outer ``bert.`` prefix. The conversion handles either.)
    """
    rest = hf_key
    if rest.startswith("bert."):
        rest = rest[len("bert."):]

    # Embeddings
    if rest == "embeddings.word_embeddings.weight":
        return "word_embeddings.weight"
    if rest == "embeddings.position_embeddings.weight":
        return "position_embeddings.weight"
    if rest == "embeddings.token_type_embeddings.weight":
        return "token_type_embeddings.weight"
    if rest == "embeddings.LayerNorm.weight":
        return "embeddings.LayerNorm.weight"
    if rest == "embeddings.LayerNorm.bias":
        return "embeddings.LayerNorm.bias"

    # Final encoder norm
    if rest == "encoder.LayerNorm.weight":
        return "encoder.LayerNorm.weight"
    if rest == "encoder.LayerNorm.bias":
        return "encoder.LayerNorm.bias"

    # Encoder layers
    if rest.startswith("encoder.layer."):
        # encoder.layer.{i}.attention.self.query.weight
        # encoder.layer.{i}.attention.self.key.weight
        # encoder.layer.{i}.attention.self.value.weight
        # encoder.layer.{i}.attention.output.dense.weight
        # encoder.layer.{i}.attention.output.LayerNorm.weight
        # encoder.layer.{i}.attention.output.LayerNorm.bias
        # encoder.layer.{i}.intermediate.dense.weight
        # encoder.layer.{i}.intermediate.dense.bias
        # encoder.layer.{i}.output.dense.weight
        # encoder.layer.{i}.output.dense.bias
        # encoder.layer.{i}.output.LayerNorm.weight
        # encoder.layer.{i}.output.LayerNorm.bias
        parts = rest.split(".")
        # encoder / layer / {i} / <block> / <sub> [/ <sub>]
        i = parts[2]
        block = parts[3]
        if block == "attention":
            if len(parts) >= 5 and parts[4] == "self":
                kind = parts[5]  # query/key/value
                if kind in ("query", "key", "value"):
                    bias = len(parts) == 7 and parts[6] == "bias"
                    suffix = ".bias" if bias else ".weight"
                    proj = {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[kind]
                    return f"encoder.layer.{i}.attention.{proj}{suffix}"
            if len(parts) >= 5 and parts[4] == "output":
                if parts[5] == "dense":
                    bias = len(parts) == 7 and parts[6] == "bias"
                    suffix = ".bias" if bias else ".weight"
                    return f"encoder.layer.{i}.attention.o_proj{suffix}"
                if parts[5] == "LayerNorm":
                    suffix = "." + parts[6]  # weight / bias
                    return f"encoder.layer.{i}.attention.LayerNorm{suffix}"
        if block == "intermediate":
            if parts[4] == "dense":
                bias = len(parts) == 6 and parts[5] == "bias"
                suffix = ".bias" if bias else ".weight"
                return f"encoder.layer.{i}.ffn.lin1{suffix}"
        if block == "output":
            if parts[4] == "dense":
                bias = len(parts) == 6 and parts[5] == "bias"
                suffix = ".bias" if bias else ".weight"
                return f"encoder.layer.{i}.ffn.lin2{suffix}"
            if parts[4] == "LayerNorm":
                suffix = "." + parts[5]
                return f"encoder.layer.{i}.ffn.LayerNorm{suffix}"

    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Convert BGE weights to MLX format")
    p.add_argument(
        "--src",
        default="BAAI/bge-base-en-v1.5",
        help="Hugging Face model id (BGE-base-en-v1.5, not the Xenova fork)",
    )
    p.add_argument(
        "--dst",
        default=str(Path.home() / ".cache" / "atlas" / "models" / "bge-base-en-v1.5-mlx"),
        help="Output directory for .npy weight files",
    )
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("  PyTorch is required for weight conversion. Install with:", file=sys.stderr)
        print("    uv pip install torch", file=sys.stderr)
        return 1

    try:
        from transformers import AutoModel
    except ImportError:
        print("  transformers is required. Install with:", file=sys.stderr)
        print("    uv pip install transformers", file=sys.stderr)
        return 1

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    print(f"  Loading {args.src} (PyTorch state dict)...")
    model = AutoModel.from_pretrained(args.src)
    state = model.state_dict()

    converted = 0
    skipped = 0
    for hf_key, tensor in state.items():
        mlx_key = _hf_to_mlx(hf_key)
        if mlx_key is None:
            skipped += 1
            continue
        arr = tensor.detach().cpu().numpy()
        np.save(dst / f"{mlx_key}.npy", arr)
        converted += 1

    # Also save a small config so consumers can verify the conversion.
    config = {
        "source_model": args.src,
        "format": "mlx-weights-v1",
        "n_tensors": converted,
        "n_skipped": skipped,
        "dim": 768,
        "n_layers": 12,
        "n_heads": 12,
        "ff_dim": 3072,
        "max_seq": 512,
        "vocab_size": 30522,
    }
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    print(f"  Wrote {converted} tensors to {dst}")
    print(f"  Skipped {skipped} non-parameter buffers")
    print(f"  Wrote {dst / 'config.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
