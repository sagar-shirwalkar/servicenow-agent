import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from agent import run_agent # Import the agent from Step 5


#
# Since we are using Zed and Opencode, the modern standard for integrating local agents 
# without a GUI is the Model Context Protocol (MCP). 
# Zed natively supports MCP servers.
# We will wrap our agent in a lightweight MCP server using the official Python SDK.
# 


app = Server("servicenow-agent")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="execute_servicenow_agent",
            description="Runs the ServiceNow Plan/Build agent to generate code or OpenAPI specs.",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The coding or API task to accomplish."}
                },
                "required": ["task"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "execute_servicenow_agent":
        task = arguments["task"]
        # Run the agent synchronously (or use asyncio.to_thread for production)
        result = run_agent(task) 
        return [TextContent(type="text", text=result)]
    raise ValueError(f"Tool not found: {name}")

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
