"""Fine-tuning pipeline for a local ServiceNow model (planned).

Atlas v0.1 shipped an Unsloth QLoRA script that was never actually
run in the v0.1 → v0.2 transition (it was a system-prompt override
on a base Qwen the whole time). The new architecture makes the
fine-tuning question genuinely interesting again: now that we have
a portable RAG surface, a small local model can use it as an
external memory and a fine-tune can specialize the model's
*output style* (code conventions, citation habits, ServiceNow
idiom) without trying to bake the entire knowledge base into the
weights.

This module will host:

  * Dataset curation from the same RAG bundle used for retrieval
    (chunks.parquet is already the perfect input).
  * QLoRA adapters for Qwen 2.5 / 3.5 Coder, with evaluation
    against the same tool-calling contracts the MCP servers use.
  * GGUF export to plug into Ollama so the local model can sit
    alongside the RAG server in a fully-offline IDE setup.

Target: Apple Silicon via MLX, optionally NVIDIA via Unsloth.
Until then, the module is intentionally empty.
"""
