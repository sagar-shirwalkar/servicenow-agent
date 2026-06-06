"""Reasoning agent that consumes the Atlas MCP servers (planned).

The original prototype in this project used a hand-rolled ReAct loop
against Ollama. The new design delegates reasoning to whichever
agent the user already runs in their IDE (opencode, Zed, Claude
Desktop) and exposes the Atlas servers as tools. That makes the
agent itself an open question — any model can be the brain, Atlas
is the memory.

This module will eventually host:

  * A thin local agent that wraps the two MCP servers for
    command-line use (``atlas-agent "find docs about SLA breaches"``).
  * Pre-built tool-use prompt templates tuned for ServiceNow
    tasks (OpenAPI generation, GlideRecord scripting, workflow
    synthesis).
  * A planner that fans a question out to fs_server + rag_server
    in parallel and merges the results.

Until then, the module is intentionally empty. The MCP servers
are the public surface; anything that speaks MCP is a valid
client.
"""
