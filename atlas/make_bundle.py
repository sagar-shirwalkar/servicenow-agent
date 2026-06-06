"""Build a portable RAG bundle from the ServiceNowDocs repo.

End-to-end bundle build. `git pull` the docs, walk every `.md`,
chunk, embed, write `chunks.parquet` + `embeddings.f16.npy` +
`norms.f32.npy` + `model/` + `manifest.json`. Records the pinned
source SHA so re-runs are reproducible. Prints the chosen embedding
backend and the reason for the choice. Accepts
`--prefer {auto,apple,nvidia,cpu}`.

Pipeline:
  1. ``git fetch`` the pinned ServiceNowDocs branch (default:
     ``australia``) into ``--repo-path``.
  2. Walk every ``markdown/**/*.md`` file and H2-chunk it.
  3. Embed all chunks with ``Xenova/bge-base-en-v1.5`` (ONNX) in
     batches with progress and retries.
  4. Persist:
       - chunks.parquet   (text + metadata)
       - embeddings.f16.npy
       - norms.f32.npy
       - model/           (ONNX model + tokenizer)
       - manifest.json    (provenance + integrity metadata)
  5. Print a summary and a SHA256 of each artifact.

The output is a self-contained directory that any MCP client can
point ``mcp_rag_server.py --bundle <dir>`` at. No GPU, no Ollama,
no torch required at runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .chunk import chunk_file
from .embed import (
    DEFAULT_MODEL_ID,
    Embedder,
    get_embedder,
    resolve_backend,
)

DEFAULT_BRANCH = "australia"
DEFAULT_REPO_URL = "https://github.com/ServiceNow/ServiceNowDocs.git"
DEFAULT_LOCAL_PATH = "./data/servicenow-docs/ServiceNowDocs-australia"

BUNDLE_SCHEMA_VERSION = 1


def ensure_repo(repo_path: Path, repo_url: str, branch: str) -> Path:
    """Clone or update the docs repo at the pinned branch."""
    if repo_path.exists() and (repo_path / ".git").is_dir():
        print(f"  Fetching latest {branch} into {repo_path}...")
        subprocess.run(
            ["git", "-C", str(repo_path), "fetch", "origin", branch],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "reset", "--hard", f"origin/{branch}"],
            check=True,
        )
    else:
        print(f"  Cloning {repo_url}@{branch} into {repo_path}...")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                repo_url,
                str(repo_path),
            ],
            check=True,
        )
    return repo_path.resolve()


def current_sha(repo_path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def walk_markdown(repo_path: Path) -> list[Path]:
    md_root = repo_path / "markdown"
    if not md_root.is_dir():
        raise FileNotFoundError(f"Expected {md_root} to exist")
    return sorted(md_root.rglob("*.md"))


def build_chunk_dataframe(files: list[Path], repo_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for i, path in enumerate(files, 1):
        try:
            chunks = chunk_file(path, repo_root)
        except Exception as e:
            print(f"  [chunk] {path.name}: {e}")
            continue
        for j, c in enumerate(chunks):
            rows.append(
                {
                    "id": f"{c['publication']}/{c['file']}#{j}",
                    "text": c["text"],
                    "publication": c["publication"],
                    "file": c["file"],
                    "heading": c["heading"],
                    "is_code": c["is_code"],
                    "title": c["frontmatter"].get("title", ""),
                    "product_area": c["frontmatter"].get("product_area", ""),
                    "last_updated": str(c["frontmatter"].get("last_updated", "")),
                    "canonical_url": c["frontmatter"].get("canonical_url", ""),
                }
            )
        if i % 500 == 0:
            print(f"  Chunked {i}/{len(files)} files ({len(rows)} chunks so far)")
    return pd.DataFrame(rows)


def stage_model(model_dir: Path, bundle_dir: Path) -> Path:
    """Copy the ONNX model + tokenizer into the bundle.

    The Xenova repo stores the model under ``onnx/`` and tokenizer
    files at the root. We mirror that layout so the runtime can
    load it the same way it would load from Hugging Face.
    """
    target = bundle_dir / "model"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    onnx_src = model_dir / "onnx"
    if onnx_src.is_dir():
        shutil.copytree(onnx_src, target / "onnx")
    for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = model_dir / name
        if src.exists():
            shutil.copy2(src, target / name)
    return target


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
    norms = np.linalg.norm(emb_to_save.astype(np.float32), axis=1).astype(np.float32)
    norms_path = output_dir / "norms.f32.npy"
    np.save(norms_path, norms)
    return emb_path, norms_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(
    bundle_dir: Path,
    source_repo: str,
    source_branch: str,
    source_sha: str,
    chunk_count: int,
    model_id: str,
    embedding_backend: str = "",
    embedding_active_provider: str = "",
) -> Path:
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "source_repo": source_repo,
        "source_branch": source_branch,
        "source_sha": source_sha,
        "source_published": None,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": chunk_count,
        "embedding_model": model_id,
        "embedding_dim": 768,
        "artifacts": {
            "chunks": "chunks.parquet",
            "embeddings": "embeddings.f16.npy",
            "norms": "norms.f32.npy",
            "model_dir": "model/",
        },
    }
    if embedding_backend:
        manifest["embedding_backend"] = embedding_backend
    if embedding_active_provider:
        manifest["embedding_active_provider"] = embedding_active_provider
    for key, rel in list(manifest["artifacts"].items()):
        if key == "model_dir":
            continue
        p = bundle_dir / rel
        if p.exists():
            manifest["artifacts"][f"{key}_sha256"] = sha256_file(p)
    out = bundle_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the ServiceNow RAG bundle")
    p.add_argument("--repo-path", default=DEFAULT_LOCAL_PATH, help="Local path to ServiceNowDocs clone")
    p.add_argument("--repo-url", default=DEFAULT_REPO_URL, help="ServiceNowDocs git URL")
    p.add_argument("--branch", default=DEFAULT_BRANCH, help="ServiceNowDocs branch (e.g. australia)")
    p.add_argument("--output", required=True, type=Path, help="Output bundle directory")
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id or local path")
    p.add_argument("--limit", type=int, default=0, help="Limit number of files (for smoke tests)")
    p.add_argument("--skip-embed", action="store_true", help="Skip embedding (for chunk-only smoke tests)")
    p.add_argument(
        "--prefer",
        choices=["auto", "apple", "nvidia", "cpu"],
        default="auto",
        help="Embedding backend preference: apple=MLX, nvidia=CUDA, cpu=ONNX+CPU, auto=probe",
    )
    return p.parse_args()


def _run() -> int:
    args = parse_args()
    repo_path = ensure_repo(Path(args.repo_path), args.repo_url, args.branch)
    sha = current_sha(repo_path)
    print(f"  Pinned to {sha}")

    files = walk_markdown(repo_path)
    if args.limit:
        files = files[: args.limit]
    print(f"  Found {len(files)} markdown files")

    df = build_chunk_dataframe(files, repo_path)
    df = df[df["text"].str.strip().astype(bool)]
    print(f"  Built {len(df)} chunks")

    args.output.mkdir(parents=True, exist_ok=True)
    chunks_path = args.output / "chunks.parquet"
    df.to_parquet(chunks_path, index=False, compression="snappy")
    print(f"  Wrote {chunks_path}")

    if args.skip_embed:
        print("  --skip-embed set; bundle contains chunks only")
        write_manifest(
            args.output, args.repo_url, args.branch, sha, len(df), args.model
        )
        return 0

    backend, reason = resolve_backend(args.prefer)
    print(f"  Embedding backend: {backend} ({reason})")
    print(f"  Loading model {args.model}...")
    embedder = get_embedder(args.model, prefer=args.prefer)
    print(f"  Active provider: {embedder.active_provider}")
    embeddings = embedder.embed_with_progress(df["text"].tolist())

    emb_path, norms_path = save_bundle_artifacts(embeddings, args.output)
    print(f"  Wrote {emb_path}")
    print(f"  Wrote {norms_path}")

    stage_model(embedder.resolved_dir, args.output)
    print(f"  Staged model files into {args.output / 'model'}")

    manifest_path = write_manifest(
        args.output,
        args.repo_url,
        args.branch,
        sha,
        len(df),
        args.model,
        embedding_backend=backend,
        embedding_active_provider=str(embedder.active_provider),
    )
    print(f"  Wrote {manifest_path}")

    print("\n  Bundle ready.")
    print(f"    chunks : {len(df)}")
    print(f"    dim    : {embeddings.shape[1]}")
    print(f"    path   : {args.output.resolve()}")
    return 0


def main() -> None:
    """Script entry point.

    Calls ``sys.exit(_run())`` so the return code propagates through
    both ``python -m`` and the console-script entry points defined
    in ``pyproject.toml``. Console scripts do not propagate return
    values on their own, so the ``sys.exit`` has to live here.
    """
    sys.exit(_run())


if __name__ == "__main__":
    main()
