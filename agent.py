#!/usr/bin/env python3
"""
ReAct agent with cross-topic workflow synthesis capabilities.
Uses directory-aware RAG and fine-tuned model for ServiceNow development.
"""

import ollama
import json
from rag_system import retrieve_context, retrieve_cross_topic

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_servicenow_docs",
            "description": "Search ServiceNow documentation. Use for single-topic queries.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string", "enum": ["api", "code", "docs"]},
                    "topic": {"type": "string", "description": "Specific topic like 'it service management'"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function", 
        "function": {
            "name": "synthesize_cross_topic_workflow",
            "description": "Synthesize workflows spanning multiple ServiceNow topics. Use for complex planning tasks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {"type": "string", "description": "The workflow goal to achieve"},
                    "topics": {"type": "array", "items": {"type": "string"}, "description": "List of topics to synthesize"}
                },
                "required": ["goal", "topics"]
            }
        }
    }
]

def run_agent(user_prompt: str, model: str = "servicenow-expert"):
    messages = [
        {
            "role": "system",
            "content": """You are an expert ServiceNow AI Agent for application development.
            
WORKFLOW:
1. ANALYZE: Determine if the request is single-topic or requires cross-topic synthesis.
2. RETRIEVE: Use appropriate tool to gather context.
3. SYNTHESIZE: Combine retrieved knowledge with your fine-tuned expertise.
4. OUTPUT: Generate production-ready code, OpenAPI specs, or workflow plans.

RULES:
- Always cite source topics in your response.
- For code: include error handling and follow ServiceNow best practices.
- For APIs: output valid JSON conforming to OpenAPI 3.0.
- When uncertain, retrieve more context before answering."""
        },
        {"role": "user", "content": user_prompt}
    ]
    
    while True:
        # --- ADD HISTORY PRUNING HERE ---
        # Keep System (0), User (1), and the last 8 messages.
        # This prevents infinite context growth during complex multi-step planning.
        if len(messages) > 10:
            messages = [messages[0], messages[1]] + messages[-8:]
            
        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS,
            options={"temperature": 0.1, "num_predict": 2048, "num_ctx": 32768}
        )
        
        messages.append(response.message)
        
        if response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                func = tool_call.function
                args = json.loads(func.arguments) if isinstance(func.arguments, str) else func.arguments
                
                if func.name == "search_servicenow_docs":
                    context = retrieve_context(
                        args["query"],
                        category_filter=args.get("category"),
                        topic_filter=args.get("topic")
                    )
                    messages.append({
                        "role": "tool",
                        "content": context,
                        "name": func.name,
                        "tool_call_id": tool_call.id
                    })
                    
                elif func.name == "synthesize_cross_topic_workflow":
                    context = retrieve_cross_topic(
                        args["goal"],
                        topics=args["topics"]
                    )
                    messages.append({
                        "role": "tool", 
                        "content": context,
                        "name": func.name,
                        "tool_call_id": tool_call.id
                    })
        else:
            return response.message.content
