# ServiceNow Knowledge: Local MCP Servers for Documentation Access

Two complementary [Model Context Protocol](https://modelcontextprotocol.io)
servers that give any MCP-aware agent (Zed, opencode, Claude Desktop,
etc.) deep access to the official
[ServiceNowDocs](https://github.com/ServiceNow/ServiceNowDocs)
repository, designed to run entirely on **Apple Silicon (M-series)**.

| Server | What it is | Models at runtime |
| --- | --- | --- |
| **`mcp_fs_server.py`** | Navigable filesystem over the docs repo | None — pure Python + ripgrep |
| **`mcp_rag_server.py`** | Semantic search over a pre-built vector bundle | Tiny ONNX embedder (110M params) on Apple Neural Engine |

The RAG bundle is **built once** by the maintainer (or by CI), then
distributed as a single download. End users never embed, never chunk,
never run a vector database, never pull a model. They just install
two servers and get instant, citation-backed knowledge of ServiceNow.

---

## 1. Why two servers and not one?

A single RAG pipeline forces every consumer into a fixed retrieval
strategy and a fixed embedding model. Splitting into a filesystem
server and a RAG server buys:

- **Different strengths.** The filesystem server is unbeatable for
  "list the publications," "read this specific file," and "grep for
  the exact symbol." The RAG server is unbeatable for "find docs
  that talk about X" where the wording is fuzzy.
- **Different trust levels.** The filesystem server returns the
  verbatim markdown — no embedding can ever lie about it. The RAG
  server returns ranked candidates; the model should still verify
  with the filesystem server for anything load-bearing.
- **Different cost profiles.** The filesystem server is ~zero startup.
  The RAG server loads ~1GB of vectors at startup but answers in
  ~100ms. A smart client uses both: RAG to discover, FS to verify.
- **Different model fits.** A weak local model (9B Qwen) struggles
  with multi-hop FS navigation but does fine with RAG top-k. A
  strong external model can do either, and benefits from using both.

This also means the project no longer fine-tunes a base model. The
local 9B Qwen, the in-editor mini-models, or any frontier model can
all consume the same MCP servers and get the same knowledge.

---

## 2. Tech stack & rationale

| Component | Choice | Why |
| --- | --- | --- |
| **MCP transport** | Official Python `mcp` SDK (stdio) | The standard. Works in Zed, opencode, Claude Desktop, anything else that speaks MCP. |
| **Filesystem search** | `ripgrep` subprocess | 10-100× faster than Python `re` over 250k+ files. Single binary, well-maintained. |
| **Embedding model** | `Xenova/bge-base-en-v1.5` (ONNX) | 110M params, 768-dim, MTEB top-30, clean ONNX export. The Xenova repo ships the pre-exported ONNX graph so we don't need `torch` or `optimum` at build time. |
| **Inference runtime** | `onnxruntime` with `CoreMLExecutionProvider` | Apple's CoreML EP dispatches the embedder to the Apple Neural Engine on M-series. Falls back to CPU automatically. |
| **Vector store** | `numpy` `.npy` arrays | At 250k × 768 dims, a single matrix multiply is ~50ms on Apple Silicon. FAISS is overkill; ChromaDB is overkill and not portable. |
| **Chunk metadata store** | `pyarrow` Parquet (snappy compressed) | Columnar, compressed, zero-copy reads, industry standard. Better than SQLite for our read pattern (load once at startup). |
| **Embedding dtype** | `float16` for vectors, `float32` for norms | Cosine similarity is rank-preserving under half precision. Halves bundle size. |
| **Tokenizer** | `transformers` `AutoTokenizer` | First-class support for the BGE fast tokenizer. Runtime dep is small relative to the ONNX model. |
| **Docs source** | `ServiceNow/ServiceNowDocs` `australia` branch (Q2 2026) | The official, purpose-built LLM-friendly docs source. Branch name tracks the ServiceNow release family. |
| **Chunking** | H2-boundary sections per markdown file | Respects the docs team's deliberate structure. One H2 = one chunk. Larger sections fall back to paragraph splits. |
| **Frontmatter** | YAML parsed at chunk time | Every doc file has `title`, `product_area`, `last_updated`, `canonical_url` in frontmatter. We carry these into chunk metadata. |
| **Distribution** | GitHub Releases (per-tag) | Simple, free, has a CLI-friendly API. End users download with one command. |
| **Package management** | `uv` | Fast resolver, lockfile, virtualenv. Stays consistent across Mac/Linux CI. |
| **CI** | GitHub Actions on `ubuntu-latest` | The bundle is platform-agnostic; building on Linux is cheaper than macOS runners. Runtime Apple Silicon acceleration comes from CoreML, not the build. |

### What we explicitly do **not** use, and why

- **No ChromaDB.** Not portable (version-coupled binary sqlite).
  Bigger than the data it stores. No benefit at our scale.
- **No Ollama.** The whole point of going ONNX is to avoid pulling
  a quantized LLM and a daemon. The embedder is 110M params and
  runs in milliseconds on the ANE.
- **No `torch` / `optimum`.** The Xenova repo ships pre-exported
  ONNX. We use that. Saves ~2GB of Python deps.
- **No `markdown-it-py` AST parser.** H2 chunking is one regex.
  An AST parser is overkill when the chunk boundary is literally
  a line prefix.
- **No fine-tuning.** The local 9B was never actually fine-tuned
  in v1; the system prompt alone was doing all the work. v2
  removes the dead code path entirely.
- **No CUDA.** Apple Silicon only. ONNX's CUDA EP is intentionally
  absent from `embed.available_providers()` to keep the dep
  surface clean.

---

## 3. Project structure

```
servicenow-agent/
├── README.md                  This file
├── pyproject.toml             uv-managed deps, build/dev extras
├── .gitignore                 Ignores bundles, clones, venvs
│
├── mcp_fs_server.py           SERVER 1: filesystem MCP
├── mcp_rag_server.py          SERVER 2: RAG MCP
│
├── chunk.py                   H2-boundary chunker + frontmatter parser
├── embed.py                   ONNX embedder (batching, progress, retries)
├── make_bundle.py             Orchestrator: clone + chunk + embed + package
│
├── download.py                Fetch + verify the latest bundle
├── backup.py                  Snapshot the current bundle
├── restore.py                 Roll back to a previous snapshot
├── smoke_test.py              1-2 min end-to-end validation
│
├── .github/workflows/
│   └── build-bundle.yml       CI: builds and publishes to Releases
│
├── ServiceNowDocs-australia/  (gitignored) local clone of the docs
└── servicenow-rag-bundle/     (gitignored) local bundle directory
```

### What each file does

**`chunk.py`** — Markdown → chunks. Splits on `## ` headers, parses
YAML frontmatter, flags code-dominated chunks, falls back to
paragraph splitting for oversized sections. No AST, no regex horrors.

**`embed.py`** — Chunks → vectors. Wraps an ONNX session with a
mean-pool head, L2 normalization, batched inference, tqdm progress,
and a 3-attempt exponential-backoff retry on transient failures.
Detects Apple Silicon and prefers the CoreML execution provider.

**`make_bundle.py`** — End-to-end bundle build. `git pull` the docs,
walk every `.md`, chunk, embed, write `chunks.parquet` +
`embeddings.f16.npy` + `norms.f32.npy` + `model/` + `manifest.json`.
Records the pinned source SHA so re-runs are reproducible.

**`mcp_fs_server.py`** — Five tools: `list_publications`,
`list_files`, `read_file`, `search`, `get_release_info`. All
deterministic, all backed by file I/O and `ripgrep`. No model.
No state. Drop-in for any markdown repo.

**`mcp_rag_server.py`** — Four tools: `search_docs`, `search_code`,
`get_chunk`, `get_bundle_info`. Loads the bundle once at startup
into memory, answers queries with cosine similarity over a single
matrix multiply. Returns chunks with file paths, headings,
similarity scores, and provenance.

**`download.py`** — End-user entry point. Hits the GitHub Releases
API, downloads the asset, verifies the SHA256 of `chunks.parquet`
against the manifest, extracts in place. If a bundle already
exists at `--output`, snapshots it first via `backup.py` so a
broken new bundle can be rolled back.

**`backup.py`** / **`restore.py`** — Operational pair. `backup.py`
creates a timestamped tar.gz of the current bundle, prunes old
snapshots past `--keep`. `restore.py` lists, picks, and swaps
back. `restore.py` itself snapshots the current bundle as a
safety net before swapping, unless `--no-safety-snapshot` is set.

**`smoke_test.py`** — Builds a 20-file bundle into a tempdir,
loads it via the same code path `mcp_rag_server.py` uses, and
runs a real semantic search for "incident." Catches the obvious
failures (broken chunker, broken embedder, broken Parquet) in
~90 seconds.

**`.github/workflows/build-bundle.yml`** — Monthly cron + manual
dispatch + push-to-main. Runs `make_bundle.py` on a Linux runner,
smoke-tests, and publishes a new GitHub Release with the bundle
tarball. The release tag is `australia-YYYYMMDD` so users can
pin to a specific ServiceNow release family.

---

## 4. For users: install and use

### 4.1 Prerequisites

- **macOS on Apple Silicon (M1 / M2 / M3 / M4 / M5).** Intel Macs
  and non-Apple-Silicon machines will run the servers on CPU.
  They will not get ANE acceleration.
- **Python 3.11+** (3.12 or 3.13 recommended).
- **[`uv`](https://docs.astral.sh/uv/)** for Python env management.
- **[`ripgrep`](https://github.com/BurntSushi/ripgrep)** (`brew install ripgrep`).
- **Git.**
- ~1.5 GB of free disk for the bundle.

### 4.2 Install

```bash
git clone <this-repo-url> servicenow-agent
cd servicenow-agent
uv venv
uv pip install -e .
```

### 4.3 Get the bundle

Pick one:

**Option A: download the latest pre-built bundle (recommended)**

```bash
uv run download.py \
  --repo <owner>/servicenow-agent \
  --output ./servicenow-rag-bundle
```

This downloads, verifies, and extracts the bundle. If a previous
bundle exists, it's snapshotted into `./servicenow-rag-bundle/.backups/`
first. The most recent snapshots are kept (default 5).

Pin to a specific release:

```bash
uv run download.py \
  --repo <owner>/servicenow-agent \
  --tag australia-20260606 \
  --output ./servicenow-rag-bundle
```

**Option B: build it yourself from the docs repo**

```bash
git clone --depth 1 -b australia \
  https://github.com/ServiceNow/ServiceNowDocs.git ServiceNowDocs-australia

uv run make_bundle.py \
  --repo-path ./ServiceNowDocs-australia \
  --branch australia \
  --output ./servicenow-rag-bundle
```

This takes 20-45 minutes on Apple Silicon and downloads the BGE
ONNX model on first run (~440 MB cached afterwards).

### 4.4 Configure your IDE

#### Zed (`~/.config/zed/settings.json`)

```json
"context_servers": {
  "servicenow-fs": {
    "command": "uv",
    "args": [
      "run", "--directory", "/absolute/path/to/servicenow-agent",
      "mcp_fs_server.py",
      "--repo", "/absolute/path/to/servicenow-agent/ServiceNowDocs-australia"
    ],
    "timeout": 60
  },
  "servicenow-rag": {
    "command": "uv",
    "args": [
      "run", "--directory", "/absolute/path/to/servicenow-agent",
      "mcp_rag_server.py",
      "--bundle", "/absolute/path/to/servicenow-agent/servicenow-rag-bundle"
    ],
    "timeout": 120
  }
}
```

#### opencode (`~/.config/opencode/config.toml`)

```toml
[mcp.servers.servicenow-fs]
command = "uv"
args = [
  "run", "--directory", "/absolute/path/to/servicenow-agent",
  "mcp_fs_server.py",
  "--repo", "/absolute/path/to/servicenow-agent/ServiceNowDocs-australia"
]
timeout = 60

[mcp.servers.servicenow-rag]
command = "uv"
args = [
  "run", "--directory", "/absolute/path/to/servicenow-agent",
  "mcp_rag_server.py",
  "--bundle", "/absolute/path/to/servicenow-agent/servicenow-rag-bundle"
]
timeout = 120
```

> **Timeout tip:** the RAG server's first tool call after startup
> pays the ONNX session warmup cost (~3-5s). The filesystem server
> is instant. If your IDE times out, raise `timeout` to 120 or 300.

### 4.5 Verify

Both servers should appear in your IDE's MCP list. Try a prompt:

> "Use the servicenow-rag server to find ServiceNow documentation
> about SLA breach handling, then use servicenow-fs to read the
> most relevant file in full."

If the agent reports the search worked and cited a file path,
you're done.

A quick CLI sanity check without the IDE:

```bash
# Filesystem server (lists publications and exits)
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' \
  | uv run mcp_fs_server.py --repo ./ServiceNowDocs-australia 2>&1 \
  | head -20
```

---

## 5. For maintainers: building and releasing

### 5.1 Local build

```bash
git clone --depth 1 -b australia \
  https://github.com/ServiceNow/ServiceNowDocs.git ServiceNowDocs-australia

uv venv
uv pip install -e ".[build]"

uv run make_bundle.py \
  --repo-path ./ServiceNowDocs-australia \
  --branch australia \
  --output ./test-bundle \
  --model Xenova/bge-base-en-v1.5
```

Useful flags:
- `--limit 50` — chunk the first 50 files for a fast smoke run
- `--skip-embed` — chunk only, no embedding (debugging the chunker)
- `--model /path/to/local/model` — use a local model cache offline

The build writes a `manifest.json` with the source SHA, chunk
count, model id, and SHA256 of each artifact.

### 5.2 CI release

`.github/workflows/build-bundle.yml` runs:

1. Monthly cron (`0 6 1 * *`) to pick up new ServiceNow docs.
2. Manual dispatch for one-off rebuilds.
3. Push to `main` when build scripts change.

On each run it builds, smoke-tests, and publishes a release named
`australia-YYYYMMDD` with the bundle tarball as the sole asset.
No secrets needed.

To cut a release with a custom tag:

1. Go to **Actions → build-bundle → Run workflow**.
2. Set the `tag` input (e.g. `australia-20260606`).
3. The workflow creates a release with that tag.

### 5.3 Promoting a release

Once a release exists, users can install it with:

```bash
uv run download.py \
  --repo <owner>/servicenow-agent \
  --tag australia-20260606 \
  --output ~/servicenow-rag-bundle
```

To roll back a release, point your IDE back at an older bundle
directory and re-run `download.py` with an older tag, or:

```bash
uv run restore.py --bundle ~/servicenow-rag-bundle --list
uv run restore.py --bundle ~/servicenow-rag-bundle --from snapshot-20260601T120000Z.tar.gz
```

---

## 6. Operating

### 6.1 Update to the latest bundle

```bash
uv run download.py --repo <owner>/servicenow-agent --output ~/servicenow-rag-bundle
```

The current bundle is auto-snapshotted to
`~/servicenow-rag-bundle/.backups/` before being replaced. If the
new bundle is broken, the IDE still works against the old
directory while you sort it out.

### 6.2 List and roll back

```bash
uv run restore.py --bundle ~/servicenow-rag-bundle --list

uv run restore.py --bundle ~/servicenow-rag-bundle  # latest
uv run restore.py --bundle ~/servicenow-rag-bundle --from snapshot-20260601T120000Z.tar.gz
```

`restore.py` snapshots the current bundle as a safety net
before swapping, so you can always go back one more step.

### 6.3 Manual snapshot

```bash
uv run backup.py --bundle ~/servicenow-rag-bundle --keep 5
```

### 6.4 Refresh the docs source (filesystem server only)

```bash
cd ServiceNowDocs-australia && git pull origin australia
```

The filesystem server reads live from this directory, so a pull
is immediately visible to the agent. No restart needed.

---

## 7. Platform support & caveats

### Supported

- **Apple Silicon (M1 / M2 / M3 / M4 / M5).** All macOS versions
  with current security updates. ANE acceleration via CoreML EP.
- **Linux x86_64 + CPU.** The RAG server runs fine, just slower
  per query (~150ms vs ~20ms on M-series). The filesystem server
  is platform-agnostic.

### Not supported

- **NVIDIA GPUs / CUDA.** We do not enable ONNX's CUDA execution
  provider. Even if you `pip install onnxruntime-gpu`, the server
  will fall back to CPU. Apple Silicon is the explicit target.
- **Intel Macs.** `CoreMLExecutionProvider` is not available; the
  server will silently fall back to CPU and be slow.
- **Windows.** `mcp` and `onnxruntime` work on Windows, but we
  have not tested `tar` and `rg` paths. PRs welcome.
- **MCP clients that don't speak stdio JSON-RPC.** This is the
  only transport we support.

### Known limitations

- The embedder's `max_seq_length` is 512 tokens. Chunks larger
  than that are silently truncated by the tokenizer. The H2
  chunker rarely produces such chunks, but the `chunk._hard_split`
  fallback cuts at paragraph boundaries when it does.
- Cosine scores are not calibrated. A score of 0.7 is not
  objectively "good" — it's only meaningful relative to other
  scores from the same query. Use `min_score` to filter loosely.
- The bundle is a single branch (`australia` by default). To
  support multiple ServiceNow release families, build multiple
  bundles and switch with `--bundle`.

---

## 8. Troubleshooting

### `FileNotFoundError: No 'markdown/' directory at ...`

The filesystem server was started with `--repo` pointing at the
wrong place. The expected layout is `<repo>/markdown/<publication>/*.md`.
If you don't have a clone yet, run:

```bash
git clone --depth 1 -b australia \
  https://github.com/ServiceNow/ServiceNowDocs.git ServiceNowDocs-australia
```

### `Bundle manifest missing`

The RAG server can't find its `manifest.json`. Either
`--bundle` points at the wrong directory, or the bundle wasn't
extracted cleanly. Re-run `download.py` with `--force` (delete
the directory first).

### `ripgrep (rg) not installed`

```bash
brew install ripgrep
```

### `Context server requires timeout` in Zed/opencode

The first RAG query after server startup is slow (3-5s) due to
ONNX session warmup. Subsequent queries are ~50-100ms. Raise
the MCP `timeout` to 120 or 300 seconds.

### `CoreMLExecutionProvider not available`

You're on an Intel Mac or a non-macOS host. The RAG server
falls back to CPU. It will work, just slower.

### Search returns nothing useful

1. Run `get_bundle_info` to confirm the bundle is loaded.
2. Try `search_docs` with a broader query.
3. Use `mcp_fs_server.search` to grep for exact terms.
4. Drop `min_score` to 0.0 to see all candidates.

### Build OOMs on Linux CI

The default GitHub Actions runner has 7 GB. The build peaks
around 4-5 GB. If you fork the workflow on a smaller runner,
add `model: large` (16 GB) or set `swap`. Apple Silicon builds
have plenty of unified memory and rarely hit this.

---

## 9. Development

### Run the smoke test

```bash
uv run smoke_test.py
```

Clones aren't required — it uses the local `ServiceNowDocs-australia`
clone. Builds a 20-file test bundle, loads it, runs a search.

### Add a new tool

Both servers are intentionally minimal. To add a tool:

1. Add a `Tool(...)` entry to the `list_tools` decorator in
   `mcp_fs_server.py` or `mcp_rag_server.py`.
2. Add a handler branch in `call_tool`.
3. Keep the handler synchronous (it's already running in the
   async event loop; `asyncio.to_thread` is fine for blocking I/O).

### Add a new filter to the RAG search

`mcp_rag_server.Bundle.search` builds a `mask` from the optional
filters. Add a new filter by extending the schema, accepting it
in `search()`, and adding a `mask &= ...` line.

### Change the embedder

Edit `embed.DEFAULT_MODEL_ID` and `embed.EMBEDDING_DIM`. The bundle
build and the runtime server read both. Be aware that switching
embedding models invalidates all existing bundles.

### Change the chunker

Edit `chunk.py`. The schema is documented in the module docstring.
Run `smoke_test.py` after changes.

---

## 10. License

This project is licensed under the terms in [LICENSE](LICENSE).
The ServiceNowDocs content is governed by the upstream license
(see `ServiceNowDocs-australia/LICENSE` after cloning).
