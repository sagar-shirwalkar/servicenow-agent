import ollama
import json
from rag_system import retrieve_context


#
# Agent Orchestration (Plan, Build, OpenAPI)
# We will build a ReAct (Reason + Act) agent using Ollama's native Tool Calling feature. 
# Qwen 2.5 and Llama 3.1 support this natively, eliminating the need for heavy frameworks like LangChain.
# 


# Define Tools for the Agent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_servicenow_docs",
            "description": "Search the ServiceNow documentation for APIs, code patterns, or concepts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_openapi_json",
            "description": "Validates if a generated JSON payload conforms to standard REST/OpenAPI structures.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "string", "description": "The JSON string to validate."}
                },
                "required": ["payload"]
            }
        }
    }
]

def run_agent(user_prompt: str):
    messages = [
        {
            "role": "system", 
            "content": "You are an expert ServiceNow AI Agent. Your workflow: 1. PLAN your approach. 2. USE TOOLS to gather context. 3. BUILD the final code or OpenAPI spec. Always output valid JSON or Markdown code blocks."
        },
        {"role": "user", "content": user_prompt}
    ]
    
    # Agent Loop
    while True:
        response = ollama.chat(
            model="servicenow-expert", # Our fine-tuned model
            messages=messages,
            tools=TOOLS
        )
        
        messages.append(response.message)
        
        # If the model decides to use a tool
        if response.message.tool_calls:
            for tool in response.message.tool_calls:
                if tool.function.name == "search_servicenow_docs":
                    query = tool.function.arguments['query']
                    context = retrieve_context(query)
                    # Feed tool result back to the model
                    messages.append({
                        "role": "tool",
                        "content": context,
                        "name": tool.function.name
                    })
                elif tool.function.name == "validate_openapi_json":
                    # Simple mock validation logic
                    try:
                        json.loads(tool.function.arguments['payload'])
                        result = "Valid JSON"
                    except:
                        result = "Invalid JSON"
                    messages.append({"role": "tool", "content": result, "name": tool.function.name})
        else:
            # No tool calls, the agent has generated its final answer
            return response.message.content

# Example Usage:
# print(run_agent("Write a Python script to fetch all active incidents using the Table API."))
