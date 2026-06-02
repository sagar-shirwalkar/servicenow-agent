# ServiceNow Local AI Agent: Hybrid RAG & Fine-Tuning Architecture

A high-efficiency, local-first AI agent designed specifically for ServiceNow application development. This system combines Retrieval-Augmented Generation (RAG) with targeted model fine-tuning to provide expert-level coding, OpenAPI specification generation, and cross-topic workflow synthesis without relying on cloud-based APIs.

## Objectives and Methodology

### Objectives
The primary objective of this project is to create a general-purpose, domain-specialized AI agent capable of planning and building complex ServiceNow applications. It is engineered to run entirely on local consumer hardware (e.g., Apple Silicon M-series or NVIDIA consumer GPUs) using quantized ~9B parameter models, ensuring data privacy, zero API costs, and offline capability.

### Methodology
The architecture employs a **Hybrid Intelligence Model** that synergizes two distinct learning paradigms:

1. **AST-Aware Data Ingestion:** 
   Unlike naive text-splitters that destroy code blocks and API schemas, our ingestion pipeline (`data_pipeline.py`) uses an Abstract Syntax Tree (AST) parser (`markdown-it-py`). It respects Markdown hierarchy, keeping narrative explanations logically bound to their corresponding code snippets and OpenAPI schemas. It utilizes a rolling-window accumulator to guarantee chunk sizes remain within the embedding model's token limits while preserving semantic continuity.
   
2. **Retrieval-Augmented Generation (RAG):** 
   Serves as the agent's dynamic external memory. By indexing the entirety of the `ServiceNowDocs-australia` repository into a local vector database, the agent can retrieve precise, up-to-date factual information across all ServiceNow domains (ITSM, Security Ops, Cloud Observability, etc.) at inference time.

3. **Unified Fine-Tuning (Mixture of Experts Approximation):** 
   Instead of routing to multiple specialized models, we approximate a Mixture-of-Experts (MoE) by fine-tuning a single base model on a blended dataset of API examples, coding patterns, and conceptual documentation. This ingrains ServiceNow syntax, GlideRecord patterns, and REST conventions directly into the model's weights.

4. **ReAct Orchestration & Cross-Topic Synthesis:** 
   The agent operates on a Reason + Act (ReAct) loop. It is equipped with specialized tools to search the vector database, synthesize cross-topic workflows (e.g., linking Account Lifecycle Events with Security Operations), and validate OpenAPI JSON payloads. 

## Tech Stack

* **Base LLM:** Qwen 2.5 Coder 7B (or Qwen 3.5 9B MLX for Apple Silicon) - Chosen for superior JSON/OpenAPI handling and code generation.
* **Inference Engine:** Ollama (Local quantized GGUF/MLX serving).
* **Vector Database:** ChromaDB (Persistent, local, in-process).
* **Embedding Model:** `nomic-embed-text` (Served locally via Ollama).
* **Fine-Tuning:** Unsloth (QLoRA 4-bit parameter-efficient fine-tuning).
* **Data Parsing:** `markdown-it-py` (AST Markdown parsing).
* **Agent Orchestration:** Native Ollama Tool Calling (ReAct loop).
* **IDE Integration:** Model Context Protocol (MCP) for seamless integration with Zed and Opencode.
* **Package Management:** `uv` (Astral's ultra-fast Python package manager).

## Prerequisites

### Hardware Requirements
* **Apple Silicon (Mac):** M4/M5 with at least 16GB Unified Memory (Recommended for MLX models).
* **NVIDIA GPU:** RTX 3060 (12GB) or better for QLoRA fine-tuning and GGUF inference.
* **RAM:** 16GB minimum (32GB recommended for indexing 250k+ chunks).

### Software Requirements
* **Python:** 3.11 or higher.
* **Ollama:** Installed and running locally (`ollama serve`).
* **Git:** For cloning the documentation repository.
* **uv:** Fast Python package installer and resolver (`curl -LsSf https://astral.sh/uv/install.sh | sh`).

## 4. Installation & Setup Instructions

### Step 1: Environment Setup
Clone this repository and initialize the Python environment using `uv`.

```bash
git clone <your-repo-url> servicenow-agent
cd servicenow-agent
uv init
uv add ollama chromadb unsloth markdown-it-py pydantic mcp
```

### Step 2: Pull Required Ollama Model
Download the embedding model and your chosen base model.

```bash
ollama pull nomic-embed-text
# For standard GGUF (NVIDIA/CPU):
ollama pull qwen2.5-coder:7b-instruct-q6_K
# OR for Apple Silicon MLX:
ollama pull qwen3.5:9b-mlx
```

### Step 3: Ingest Documentation
The ingest.py script will automatically clone the ServiceNowDocs-australia repository, parse the Markdown files using the AST pipeline, and index them into ChromaDB.

```bash
# First time run (clones repo and builds index)
uv run ingest.py

# To force a complete rebuild of the vector database later:
uv run ingest.py --force
```

Note: Indexing ~275,000 chunks takes approximately 30-45 minutes on an M5 Pro chip or modern retail NVIDIA GPU.

### Step 4: Fine-Tuning 
To create the specialized servicenow-expert model, generate the training data and run the Unsloth fine-tuning script.

```
# 1. Generate the blended training dataset
uv run prepare_dataset.py

# 2. Run QLoRA fine-tuning (Requires NVIDIA GPU or Apple Silicon with Unsloth support)
uv run finetune_unsloth.py
```

### Step 5: Create the Ollama Modelfile
Create a file named Modelfile in the root directory to configure the model's system prompt and context window.

```dockerfile
FROM qwen3.5:9b-mlx
# OR use your fine-tuned GGUF: FROM ./qwen_servicenow_gguf/unsloth.Q6_K.gguf

SYSTEM "You are an expert ServiceNow developer and API architect."
PARAMETER stop "<|im_end|>"
PARAMETER num_ctx 32768
```

Register the model with Ollama locally:

```bash
ollama create servicenow-expert -f Modelfile
```

## Running the Agent

### Standalone CLI Test
You can test the agent's ReAct loop directly from your terminal:

```bash
uv run agent.py "Write a Python script to fetch all active incidents using the Table API and include proper error handling."
```

###  IDE Integration (Zed & Opencode via MCP)
To use the agent natively inside your editor, start the MCP server and configure your IDE.

1. Test the MCP Server locally:

```bash
uv run mcp_server.py
```

2. Configure Zed AI:

Open your Zed settings (~/.config/zed/settings.json) and add the context server. Ensure you use the absolute path to your project.

```json
"context_servers": {
  "servicenow-agent": {
    "command": "uv",
    "args": ["run", "/absolute/path/to/servicenow-agent/mcp_server.py"],
    "timeout": 300
  }
}
```

3. Configure Opencode (ACP/MCP):

Add the following to your ~/.config/opencode/config.toml:

```toml
[mcp.servers.servicenow-agent]
command = "uv"
args = ["run", "/absolute/path/to/servicenow-agent/mcp_server.py"]
timeout = 300
```

## Architecture & File Structure

data_pipeline.py: AST-based Markdown parser and directory-aware categorization engine.

rag_system.py: ChromaDB vector store management, byte-level pre-filtering, and cross-topic retrieval logic.

ingest.py: Orchestration script for cloning the ServiceNow docs and triggering the parse/index pipeline.

agent.py: The core ReAct loop utilizing Ollama's native tool calling for planning and execution.

mcp_server.py: Model Context Protocol wrapper for seamless IDE integration.

prepare_dataset.py / finetune_unsloth.py: Data curation and QLoRA training scripts.

## Troubleshooting

### ResponseError: the input length exceeds the context length (status code: 400)

Cause: The RAG payload or chat history exceeded the model's num_ctx.
Fix: Ensure PARAMETER num_ctx 32768 is set in your Modelfile and passed in the options dictionary in agent.py. The rag_system.py includes a hard 15,000-character safety truncation to prevent this.

### Context server requires timeout in Zed/Opencode

Cause: The agent's multi-step reasoning loop takes longer than the IDE's default 30-second timeout.
Fix: Increase the timeout parameter to 300 (5 minutes) in your IDE's MCP/ACP configuration JSON/TOML.

### No chunks extracted! during ingestion

Cause: The local repository path is incorrect or the australia branch failed to clone.
Fix: Delete the ./ServiceNowDocs-australia folder and re-run uv run ingest.py --force.
