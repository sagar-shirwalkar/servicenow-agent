# ServiceNow Atlas

*Local-first AI knowledge layer for ServiceNow development.*

Atlas is a local-first knowledge layer for ServiceNow development. Two
[Model Context Protocol](https://modelcontextprotocol.io) servers expose
the entire [ServiceNowDocs](https://github.com/ServiceNow/ServiceNowDocs)
repository to any AI agent — a navigable filesystem for verbatim
citations, and a portable RAG for semantic search. Runs entirely on
Apple Silicon. No cloud APIs, no model training, no data leaving your
machine.

The RAG bundle is **built once** by the maintainer (or by CI), then
distributed as a single download. End users never embed, never chunk,
never run a vector database, never pull a model. They install two
servers and get instant, citation-backed knowledge of ServiceNow.

> **v0.3 (current) — layered embedding backend.** The RAG server now
> picks the best inference runtime for the host: Apple MLX on M-series
> (1-2 ms/query, no ONNX bridge), ONNX Runtime + CUDA on NVIDIA Linux
> boxes, ONNX Runtime + CPU everywhere else. The bundle itself is
> backend-agnostic — only the inference runtime differs. Use
> `atlas-doctor` to see which backend will be chosen and why.
> The original ONNX+CPU code path is the portable floor; MLX and CUDA
> are optional add-ons (`uv sync --extra mlx` or `--extra gpu`).

---

## Table of contents

1. [Why two servers and not one?](#1-why-two-servers-and-not-one)
2. [Tech stack & rationale](#2-tech-stack--rationale)
3. [Project structure](#3-project-structure)
4. [For users: install and use](#4-for-users-install-and-use)
5. [For maintainers: building and releasing](#5-for-maintainers-building-and-releasing)
6. [Operating](#6-operating)
7. [Roadmap](#7-roadmap)
8. [Platform support & caveats](#8-platform-support--caveats)
9. [Troubleshooting](#9-troubleshooting)
10. [Development](#10-development)
11. [License](#11-license)

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
  The RAG server loads ~1 GB of vectors at startup but answers in
  ~100 ms. A smart client uses both: RAG to discover, FS to verify.
- **Different model fits.** A weak local model (9 B Qwen) struggles
  with multi-hop FS navigation but does fine with RAG top-k. A
  strong external model can do either, and benefits from using both.

This also means the project no longer fine-tunes a base model. The
local 9 B Qwen, the in-editor mini-models, and any frontier model can
all consume the same MCP servers and get the same knowledge.

---

## 2. Tech stack & rationale

| Component | Choice | Why |
| --- | --- | --- |
| **MCP transport** | Official Python `mcp` SDK (stdio) | The standard. Works in Zed, opencode, Claude Desktop, anything else that speaks MCP. |
| **Filesystem search** | `ripgrep` subprocess | 10-100× faster than Python `re` over 46 k+ files. Single binary, well-maintained. |
| **Embedding model** | `Xenova/bge-base-en-v1.5` (ONNX) / `BAAI/bge-base-en-v1.5` (PyTorch) | 110 M params, 768-dim, MTEB top-30. The Xenova repo ships the pre-exported ONNX graph; the PyTorch source is what we hand-convert to MLX. |
| **Inference runtime** | Layered: MLX → ONNX+CUDA → ONNX+CPU | Apple Silicon gets MLX (1-2 ms/query, 5-10× faster than CPU at build time, uses ANE/GPU directly without the ONNX→CoreML bridge). Linux with NVIDIA gets ONNX+CUDA (`--extra gpu`). Every other host gets ONNX+CPU as the portable floor. The choice is recorded in the bundle's `manifest.json` for reproducibility. See [Backend selection](#backend-selection) below. |
| **Vector store** | `numpy` `.npy` arrays | At 250 k × 768 dims, a single matrix multiply is ~50 ms on Apple Silicon. FAISS is overkill; ChromaDB is overkill and not portable. |
| **Chunk metadata store** | `pyarrow` Parquet (snappy compressed) | Columnar, compressed, zero-copy reads, industry standard. Better than SQLite for our read pattern (load once at startup). |
| **Embedding dtype** | `float16` for vectors, `float32` for norms | Cosine similarity is rank-preserving under half precision. Halves bundle size. |
| **Tokenizer** | `transformers` `AutoTokenizer` | First-class support for the BGE fast tokenizer. Runtime dep is small relative to the ONNX model. |
| **Docs source** | `ServiceNow/ServiceNowDocs` `australia` branch (Q2 2026) | The official, purpose-built LLM-friendly docs source. Branch name tracks the ServiceNow release family. |
| **Chunking** | H2-boundary sections per markdown file | Respects the docs team's deliberate structure. One H2 = one chunk. Larger sections fall back to paragraph splits. |
| **Frontmatter** | YAML parsed at chunk time | Every doc file has `title`, `product_area`, `last_updated`, `canonical_url` in frontmatter. We carry these into chunk metadata. |
| **Distribution** | GitHub Releases (per-tag) | Simple, free, has a CLI-friendly API. End users download with one command. |
| **Package management** | `uv` | Fast resolver, lockfile, virtualenv, build system. Stays consistent across Mac/Linux CI. |

---

## 3. Project structure

```
servicenow-atlas/
├── README.md                  This file
├── pyproject.toml             uv-managed deps, console-script entry points
├── .gitignore
├── .python-version
├── LICENSE
│
├── atlas/                     Python package (importable as `atlas`)
│   ├── __init__.py            Version + package docstring
│   ├── chunk.py               H2-boundary chunker + frontmatter parser
│   ├── embed/                 Embedding backends (factory pattern)
│   │   ├── base.py            ABC, factory, resolve_backend, mean_pool, l2_normalize
│   │   ├── onnx.py            OnnxEmbedder (portable, ONNX+CPU)
│   │   └── mlx.py             MlxEmbedder + hand-rolled BGE (Apple Silicon)
│   ├── fs_server.py           Filesystem MCP server
│   ├── rag_server.py          RAG MCP server (auto-selects backend)
│   ├── make_bundle.py         Build orchestrator (auto-selects backend)
│   ├── download.py            Download + verify bundle from Releases
│   ├── backup.py              Snapshot the current bundle
│   ├── restore.py             Roll back to a previous snapshot
│   ├── smoke_test.py          1-2 min end-to-end validation
│   ├── doctor.py              Diagnose installation + probe all backends
│   ├── agent.py               [planned] Reasoning agent over the MCP servers
│   └── training.py            [planned] Fine-tuning pipeline
│
├── tools/
│   └── convert_bge_to_mlx.py  One-time HF→MLX weight conversion (maintainers)
│
├── data/                      Runtime data (gitignored, see .gitignore)
│   ├── .gitkeep               Keeps the directory in git
│   ├── servicenow-docs/       Local clone of the docs (gitignored)
│   │   └── ServiceNowDocs-australia/
│   └── rag-bundle/            Pre-built RAG bundle (gitignored)
│
├── tests/                     [planned] Unit tests
│
└── .github/workflows/
    └── build-bundle.yml       Monthly CI build + GitHub Release
```

---

## 4. For users: install and use

### 4.1 Prerequisites

- **Python 3.11+** (3.12 or 3.13 recommended).
- **[`uv`](https://docs.astral.sh/uv/)** for Python env management.
- **[`ripgrep`](https://github.com/BurntSushi/ripgrep)** (`brew install ripgrep`).
- **Git.**
- ~1.5 GB of free disk for the bundle.

> **Platforms.** Atlas runs on three classes of host, each picking a
> different embedding backend by default:
>
> - **Apple Silicon (M1 / M2 / M3 / M4 / M5).** Default backend is
>   [MLX](https://github.com/ml-explore/mlx), which talks to the
>   Apple Neural Engine directly. ~1-2 ms per query. Install with
>   `uv sync --extra mlx`. Intel Macs fall back to the ONNX+CPU
>   floor; they run, just slower.
> - **Linux x86_64 with an NVIDIA GPU.** Default backend is ONNX
>   Runtime + CUDA. ~1-2 ms per query. Install with
>   `uv sync --extra gpu`. v0.3 added this path; pre-v0.3
>   explicitly skipped CUDA, which is why older notes say
>   "no NVIDIA."
> - **Anything else** (Linux/Windows without a GPU, Intel Mac, etc.)
>   falls back to the portable ONNX+CPU floor. ~10-15 ms per
>   query on a modern CPU. Always installed by default; no
>   extra required.
>
> `atlas-doctor` will tell you which backend is selected on your
> machine and why.

### 4.2 Install

```bash
git clone <this-repo-url> servicenow-atlas
cd servicenow-atlas
uv sync
```

`uv sync` resolves the lockfile, builds the package, and installs
the eight `atlas-*` console scripts into the project venv.

**Optional extras** (pick what your machine can use):

```bash
# Apple Silicon: hand-rolled MLX embedder, 5-10x faster than CPU
uv sync --extra mlx

# Linux with NVIDIA GPU: ONNX Runtime with CUDA
uv sync --extra gpu

# Both
uv sync --extra mlx --extra gpu
```

The ONNX+CPU floor needs no extra; it's installed by `uv sync` by
default. The `atlas-doctor` command will tell you which backends
are available and which one will be used by default — run it any
time you want to know what's actually wired up.

> **Re-applying extras on subsequent `uv sync`.** `uv sync`
> synchronizes the venv to match the lockfile plus whichever extras
> you specify on the command line. If you later run plain
> `uv sync` (e.g. to pick up a dep change), the MLX / GPU packages
> will be **silently removed** because they were installed under
> `--extra mlx` / `--extra gpu` and aren't in the default set. To
> keep them, re-run with the same flags:
>
> ```bash
> uv sync --extra mlx --extra gpu   # whatever you had before
> ```
>
> After a sync, `atlas-doctor` will tell you immediately if a
> previously-installed backend is now MISS.

### Backend selection

The RAG server picks an inference backend at startup, in this order:

1. The `--prefer` CLI argument (`auto` / `apple` / `nvidia` / `cpu`).
2. The `ATLAS_EMBED_BACKEND` environment variable (same values).
3. The `~/.config/atlas.toml` file, in the form:
   ```toml
   [backend]
   prefer = "apple"
   ```
4. Auto-detect: MLX on Apple Silicon, onnx-gpu on NVIDIA, else
   onnx-cpu.

The bundle itself is backend-agnostic — the build writes
`embeddings.f16.npy` (vectors only, no model in the data). A bundle
built with MLX is identical to a bundle built with ONNX+CPU, so you
can switch backends between build time and run time without
rebuilding. The choice is recorded in the build log and stderr, and
`atlas-doctor` will tell you what was selected for a given bundle.

Override examples:

```bash
# Always MLX, even on a machine that would default to CUDA
uv run atlas-rag --prefer apple

# Force the portable floor (useful for debugging MLX issues)
ATLAS_EMBED_BACKEND=cpu uv run atlas-rag
```

### 4.3 Get the docs source

```bash
git clone --depth 1 -b australia \
  https://github.com/ServiceNow/ServiceNowDocs.git \
  data/servicenow-docs/ServiceNowDocs-australia
```

This drops ~46 000 markdown files into `data/servicenow-docs/ServiceNowDocs-australia/`.
The trailing `ServiceNowDocs-australia` is the natural form of
`git clone` into a non-empty target dir.

### 4.4 Get the bundle

**Option A: download the latest pre-built bundle (recommended)**

```bash
uv run atlas-download \
  --repo <owner>/servicenow-atlas \
  --output ./data/rag-bundle
```

This downloads, verifies, and extracts the bundle. If a previous
bundle exists, it's snapshotted into `./data/rag-bundle/.backups/`
first. The most recent snapshots are kept (default 5).

Pin to a specific release:

```bash
uv run atlas-download \
  --repo <owner>/servicenow-atlas \
  --tag australia-20260606 \
  --output ./data/rag-bundle
```

**Option B: build it yourself**

```bash
uv sync --extra build
uv run atlas-build \
  --repo-path ./data/servicenow-docs/ServiceNowDocs-australia \
  --branch australia \
  --output ./data/rag-bundle
```

This takes 20-45 minutes on Apple Silicon and downloads the BGE
ONNX model on first run (~440 MB cached afterwards).

### 4.5 Configure your IDE

Both servers speak MCP over stdio, so any client that supports
[the standard](https://modelcontextprotocol.io) works: Zed,
opencode, Claude Desktop, Continue, and others.

#### Zed (`~/.config/zed/settings.json`)

```json
"context_servers": {
  "atlas-fs": {
    "command": "uv",
    "args": [
      "run", "--directory", "/absolute/path/to/servicenow-atlas",
      "atlas-fs"
    ],
    "timeout": 60
  },
  "atlas-rag": {
    "command": "uv",
    "args": [
      "run", "--directory", "/absolute/path/to/servicenow-atlas",
      "atlas-rag"
    ],
    "timeout": 120
  }
}
```

The defaults in each server already point at
`./data/servicenow-docs/ServiceNowDocs-australia` and
`./data/rag-bundle` relative to the working directory, so no
explicit `--repo` / `--bundle` flags are needed when the IDE
launches each server with `--directory` set to the project root.

#### opencode (`~/.config/opencode/config.toml`)

```toml
[mcp.servers.atlas-fs]
command = "uv"
args = [
  "run", "--directory", "/absolute/path/to/servicenow-atlas",
  "atlas-fs"
]
timeout = 60

[mcp.servers.atlas-rag]
command = "uv"
args = [
  "run", "--directory", "/absolute/path/to/servicenow-atlas",
  "atlas-rag"
]
timeout = 120
```

> **Timeout tip:** the first RAG query after server startup pays
> the ONNX session warmup cost (~3-5 s). The filesystem server is
> instant. If your IDE times out, raise `timeout` to 120 or 300.

### 4.6 Verify

Both servers should appear in your IDE's MCP list. Try a prompt:

> "Use the atlas-rag server to find ServiceNow documentation about
> SLA breach handling, then use atlas-fs to read the most relevant
> file in full."

If the agent reports the search worked and cited a file path, you're
done.

A quick CLI sanity check without the IDE:

```bash
# Confirm CLI entry points exist
uv run atlas-fs --help
uv run atlas-rag --help
```

---

## 5. For maintainers: building and releasing

### 5.0 First-time build walkthrough

A guided path for building the bundle on your own machine, from a
fresh `uv sync` to a working bundle your IDE can query. If you've
done this before and just want the command, skip to §5.1.

**Preflight.** Confirm the embedding backend you expect is actually
available:

```bash
uv sync --extra mlx        # or `--extra gpu` on Linux + NVIDIA
uv run atlas-doctor        # should report: MLX OK, selected: mlx
```

If doctor says `MLX MISS` after a sync, the optional extras were
stripped (see §4.2) — re-run with the same flags you used originally.

**Smoke build (2 minutes).** Test the pipeline with a tiny bundle
before committing to a full build:

```bash
uv run atlas-build \
  --limit 100 \
  --output /tmp/atlas-smoke \
  --prefer apple
```

This uses your existing `data/servicenow-docs/` clone, walks the
first 100 files, chunks them, embeds them. With MLX on Apple
Silicon the embedding step is ~1-2 seconds; with ONNX+CPU ~30
seconds. Free to interrupt with Ctrl-C; the next run redoes the work.

**Sanity-check the smoke bundle:**

```bash
.venv/bin/python -c "
from atlas.rag_server import Bundle
b = Bundle('/tmp/atlas-smoke', prefer='apple')
hits = b.search('ServiceNow incident handling', top_k=3)
for h in hits:
    print(f'  {h[\"score\"]:.3f}  {h[\"file\"]} :: {h[\"heading\"]}')
"
```

Using `.venv/bin/python` directly avoids the ~300 ms `uv run`
overhead. For pure JSON inspection of the manifest, `jq` is the
faster tool — see §10 for recipes.

Plausible scores are 0.5-0.7 for short queries, higher for exact
matches. If you see a tokenizer error, your venv is missing
`sentencepiece` (see §9).

**Full build (15-30 minutes):**

```bash
uv run atlas-build \
  --output ./data/rag-bundle \
  --prefer apple
```

The script `git fetch`es your existing `data/servicenow-docs/` clone
(no re-download), walks every `.md` file, and writes the bundle to
`./data/rag-bundle/`. With MLX: ~10-15 min pure embedding. With
ONNX+CPU: ~45-60 min. Total output: ~150-200 MB on disk.

**Verify and connect.** After the build, check the artifacts and
connect your IDE:

```bash
ls -lh data/rag-bundle/                # chunks.parquet, embeddings.f16.npy, norms.f32.npy, model/, manifest.json
jq . data/rag-bundle/manifest.json | head -20
```

The manifest records the source SHA, chunk count, embedding model,
and which backend was used to build. Restart your IDE so it picks
up the new bundle path; see §4.5 for Zed and opencode config.

---

### 5.1 Local build

```bash
git clone --depth 1 -b australia \
  https://github.com/ServiceNow/ServiceNowDocs.git \
  data/servicenow-docs/ServiceNowDocs-australia

uv sync --extra build

uv run atlas-build \
  --repo-path ./data/servicenow-docs/ServiceNowDocs-australia \
  --branch australia \
  --output ./data/rag-bundle \
  --model Xenova/bge-base-en-v1.5
```

Useful flags:
- `--limit 50` — chunk the first 50 files for a fast smoke run
- `--skip-embed` — chunk only, no embedding (debugging the chunker)
- `--model /path/to/local/model` — use a local model cache offline

The build writes a `manifest.json` with the source SHA, chunk count,
model id, and SHA256 of each artifact.

### 5.2 CI release

`.github/workflows/build-bundle.yml` runs:

1. Monthly cron (`0 6 1 * *`) to pick up new ServiceNow docs.
2. Manual dispatch for one-off rebuilds.
3. Push to `main` when source files change.

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
uv run atlas-download \
  --repo <owner>/servicenow-atlas \
  --tag australia-20260606 \
  --output ~/data/rag-bundle
```

To roll back a release, restore from a snapshot:

```bash
uv run atlas-restore --bundle ~/data/rag-bundle --list
uv run atlas-restore --bundle ~/data/rag-bundle --from snapshot-20260601T120000Z.tar.gz
```

---

## 6. Operating

### 6.1 Update to the latest bundle

```bash
uv run atlas-download --repo <owner>/servicenow-atlas --output ~/data/rag-bundle
```

The current bundle is auto-snapshotted to
`~/data/rag-bundle/.backups/` before being replaced. If the new
bundle is broken, the IDE still works against the old directory
while you sort it out.

### 6.2 List and roll back

```bash
uv run atlas-restore --bundle ~/data/rag-bundle --list

uv run atlas-restore --bundle ~/data/rag-bundle  # latest
uv run atlas-restore --bundle ~/data/rag-bundle --from snapshot-20260601T120000Z.tar.gz
```

`atlas-restore` snapshots the current bundle as a safety net before
swapping, so you can always go back one more step.

### 6.3 Manual snapshot

```bash
uv run atlas-backup --bundle ~/data/rag-bundle --keep 5
```

### 6.4 Refresh the docs source (filesystem server only)

```bash
cd data/servicenow-docs/ServiceNowDocs-australia && git pull origin australia
```

The filesystem server reads live from this directory, so a pull is
immediately visible to the agent. No restart needed.

### 6.5 Diagnose with `atlas-doctor`

```bash
uv run atlas-doctor
uv run atlas-doctor --bundle ~/data/rag-bundle
uv run atlas-doctor --refresh
uv run atlas-doctor --json
```

Output (short form):

```
Platform       : Darwin 25.5.0 (arm64)
Python         : 3.13.5
ONNX Runtime   : 1.26.0  providers: ['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']

Backend probe:
  ONNX+CPU       OK  always available
  Apple MLX      OK   weights cached
  NVIDIA CUDA    MISS nvidia-smi not on PATH

Selected       : mlx
                Apple Silicon detected and MLX is importable

Bundle         : /Users/me/data/rag-bundle
                manifest=OK chunks=OK embeddings=OK
                52 chunks, model=Xenova/bge-base-en-v1.5 (SHA ok)
```

`atlas-doctor` is non-destructive and never blocks startup. Run
it first when something is wrong: missing weights, wrong backend
selected, bundle SHA mismatch, missing `ripgrep`, etc. The full
probe result is cached at `~/.cache/atlas/diagnosis.json` for
24h — `--refresh` ignores the cache.

---

## 7. Roadmap

### `atlas/agent.py` — the reasoning layer (planned)

What it should eventually be:

- A thin local agent that wraps the two MCP servers for command-line
  use (`atlas-agent "find docs about SLA breaches"`) for users who
  don't have an IDE with MCP support.
- Pre-built tool-use prompt templates tuned for ServiceNow tasks
  (OpenAPI generation, GlideRecord scripting, workflow synthesis).
- A planner that fans a question out to `atlas-fs` + `atlas-rag` in
  parallel and merges the results. The current v0.2 model on the
  user's side does this by hand; the planner automates it.

The MCP servers are the public surface. Anything that speaks MCP
is a valid client. `atlas/agent.py` exists for the case where the
user doesn't have one.

### `atlas/training.py` — the fine-tuning pipeline (planned)

What it should eventually be:

- Dataset curation from the same RAG bundle used for retrieval
  (`chunks.parquet` is already the perfect input).
- QLoRA adapters for Qwen 2.5 / 3.5 Coder, with evaluation against
  the same tool-calling contracts the MCP servers use.
- GGUF export to plug into Ollama so the local model can sit
  alongside the RAG server in a fully-offline IDE setup.

Target: Apple Silicon via MLX, optionally NVIDIA via Unsloth.

The interesting shift from v0.1: now that we have a portable RAG
surface, a small local model can use it as an external memory and a
fine-tune can specialize the model's *output style* (code
conventions, citation habits, ServiceNow idiom) without trying to
bake the entire knowledge base into the weights.

### `tests/` — unit tests (planned)

A `tests/` directory alongside `atlas/` for pytest-based tests.
Right now we have `atlas-smoke` which is an end-to-end check, but
no unit-level coverage of the chunker, embedder, or the MCP tool
handlers.

---

## 8. Platform support & caveats

### Supported

- **Apple Silicon (M1 / M2 / M3 / M4 / M5), all macOS versions
  with current security updates.** MLX is the default backend on
  this hardware (`uv sync --extra mlx` to install). Single-query
  latency is ~1-2 ms.
- **Linux x86_64 + CPU.** The portable ONNX+CPU floor works
  everywhere. Single-query latency is ~10-15 ms on a modern CPU.
- **Linux x86_64 + NVIDIA GPU.** ONNX+CUDA via
  `uv sync --extra gpu`. Single-query latency is ~1-2 ms on a
  recent CUDA card.

### Not supported

- **Intel Macs.** Apple Silicon is the supported macOS path; on
  Intel Macs the server falls back to the ONNX+CPU floor. It
  will run, just without MLX acceleration.
- **Apple Neural Engine (via CoreML).** The BGE model is unstable
  under CoreML for long sequences (SIGKILL on second batch) and
  30-40x slower than CPU when it does run. v0.3 replaces the
  ONNX→CoreML bridge with MLX, which talks to the ANE directly
  without that instability.
- **Windows.** `mcp` and `onnxruntime` work on Windows, but we
  have not tested `tar` and `rg` paths. PRs welcome.
- **MCP clients that don't speak stdio JSON-RPC.** This is the
  only transport we support.

### Known limitations

- The embedder's `max_seq_length` is 512 tokens. Chunks larger
  than that are silently truncated by the tokenizer. The H2
  chunker rarely produces such chunks, but the
  `chunk._hard_split` fallback cuts at paragraph boundaries when
  it does.
- Cosine scores are not calibrated. A score of 0.7 is not
  objectively "good" — it's only meaningful relative to other
  scores from the same query. Use `min_score` to filter loosely.
- The bundle is a single branch (`australia` by default). To
  support multiple ServiceNow release families, build multiple
  bundles and switch with `--bundle`.

---

## 9. Troubleshooting

### `FileNotFoundError: No 'markdown/' directory at ...`

The filesystem server was started with `--repo` pointing at the
wrong place. The expected layout is
`<repo>/markdown/<publication>/*.md`. If you don't have a clone
yet:

```bash
git clone --depth 1 -b australia \
  https://github.com/ServiceNow/ServiceNowDocs.git \
  data/servicenow-docs/ServiceNowDocs-australia
```

### `Bundle manifest missing`

The RAG server can't find its `manifest.json`. Either `--bundle`
points at the wrong directory, or the bundle wasn't extracted
cleanly. Re-run `atlas-download` and let it overwrite (it'll
back up the current bundle first).

### `ripgrep (rg) not installed`

```bash
brew install ripgrep
```

### `Context server requires timeout` in Zed/opencode

The first RAG query after server startup pays a one-time warmup
cost: ~3-5 s for ONNX+CPU, ~1 s for MLX (the first call triggers
the MLX kernel compile). Subsequent queries are ~1-2 ms (MLX) or
~10-15 ms (ONNX+CPU). Raise the MCP `timeout` to 120 or 300
seconds to absorb the warmup.

### `atlas-doctor` says MLX is MISS but I have an M-series

You didn't install the optional MLX extra. Run:

```bash
uv sync --extra mlx
```

The MLX pip wheels are macOS-arm64-only and pull in `mlx-metal` as
a transitive dep. After `uv sync --extra mlx`, the
`atlas-doctor` probe will find them and MLX becomes the default
backend on Apple Silicon.

### `atlas-doctor` says MLX is OK but `weights cached` is missing

The MLX Python package is installed but the converted weights
have not been generated yet. Run (one-time per machine):

```bash
uv run --extra mlx python tools/convert_bge_to_mlx.py
```

This reads `BAAI/bge-base-en-v1.5` from the HuggingFace cache
(pip downloads it on first use) and writes 197 `.npy` files to
`~/.cache/atlas/models/bge-base-en-v1.5-mlx/`. The conversion is
deterministic and idempotent; subsequent runs are no-ops.

### `atlas-doctor` reports `CoreMLExecutionProvider not available`

You're on an Intel Mac or a non-macOS host. The RAG server falls
back to the ONNX+CPU floor. It will work, just slower.

> Note: as of v0.3 we no longer use the CoreML execution provider
> for BGE on Apple Silicon. MLX replaces it. The provider still
> shows up in the doctor output (it ships with `onnxruntime`) but
> it is never selected.

### Search returns nothing useful

1. Run `get_bundle_info` to confirm the bundle is loaded.
2. Try `search_docs` with a broader query.
3. Use `atlas-fs` to grep for exact terms.
4. Drop `min_score` to 0.0 to see all candidates.

### Build OOMs on Linux CI

The default GitHub Actions runner has 7 GB. The build peaks around
4-5 GB. If you fork the workflow on a smaller runner, add
`runs-on: ubuntu-latest-4-cores` with 16 GB or set up swap. Apple
Silicon builds have plenty of unified memory and rarely hit this.

### Console scripts not found after `uv sync`

If `uv run atlas-fs --help` returns "command not found," the
package isn't installed. Run `uv sync` (no flags) to install the
project itself, not just its dependencies. This installs the
`[project.scripts]` entry points into the project venv.

---

## 10. Development

### Useful shell recipes

These come up often. None of them need `uv run`:

```bash
# Pretty-print the bundle manifest (no `cat`, no `uv run` overhead)
jq . data/rag-bundle/manifest.json

# If you don't have jq, fall back to stdlib (slower, but works)
.venv/bin/python -m json.tool data/rag-bundle/manifest.json | head -20

# Load a bundle in Python without going through `uv run`
.venv/bin/python -c "
from atlas.rag_server import Bundle
b = Bundle('data/rag-bundle', prefer='apple')
for h in b.search('change request', top_k=3):
    print(f'{h[\"score\"]:.3f}  {h[\"file\"]}')
"
```

Install `jq` once (`brew install jq` on macOS, `apt install jq` on
Linux) and the first command is the one you'll reach for every time.

The pattern is: once `uv sync` is done, the venv's Python at
`.venv/bin/python` starts in ~30 ms with no project detection.
Reserve `uv run` for entry points (`atlas-build`, `atlas-rag`, ...)
and scripts that genuinely need the project's full import path.

### Run the smoke test

```bash
uv run atlas-smoke
```

Uses the local `data/servicenow-docs/ServiceNowDocs-australia/`
clone. Builds a 20-file test bundle, loads it, runs a search.

### Add a new tool

Both servers are intentionally minimal. To add a tool:

1. Add a `Tool(...)` entry to the `list_tools` decorator in
   `atlas/fs_server.py` or `atlas/rag_server.py`.
2. Add a handler branch in `call_tool`.
3. Keep the handler synchronous (it's already running in the
   async event loop; `asyncio.to_thread` is fine for blocking I/O).

### Add a new filter to the RAG search

`atlas.rag_server.Bundle.search` builds a `mask` from the optional
filters. Add a new filter by extending the schema, accepting it
in `search()`, and adding a `mask &= ...` line.

### Change the embedder

Edit `embed.DEFAULT_MODEL_ID` and `embed.EMBEDDING_DIM`. The bundle
build and the runtime server read both. Be aware that switching
embedding models invalidates all existing bundles.

### Change the chunker

Edit `atlas/chunk.py`. The schema is documented in the module
docstring. Run `atlas-smoke` after changes.

---

## 11. License

This project is licensed under the terms in [LICENSE](LICENSE).
The ServiceNowDocs content is governed by the upstream license
(see `data/servicenow-docs/ServiceNowDocs-australia/LICENSE` after
cloning).
