#!/usr/bin/env python3
import asyncio
import sys
import os
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class QdrantSearchClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
    async def connect_to_server(self, server_script_path: str):
        """Connect to the Qdrant search MCP server"""
        # Get the current Python interpreter path
        python_path = sys.executable
        
        server_params = StdioServerParameters(
            command=python_path,
            args=[server_script_path]
        )
        
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        
        await self.session.initialize()
        
        # List available tools (collections for search)
        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to Qdrant search server with tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
            
        # Log all tools for debugging
        print("\nDEBUG - All available tools:")
        for i, tool in enumerate(tools):
            print(f"Tool {i+1}: {tool.name} - {tool.description}")

    async def search(self, collection_name: str, query: str, top_k: int = 5, score_threshold: float = 0.4, filter_params=None):
        """Search a collection with the given query"""
        
        tool_name = f"search-{collection_name}"  # Changed underscore to hyphen
        arguments = {
            "query": query,
            "top_k": top_k,
            "score_threshold": score_threshold
        }
        
        if filter_params:
            arguments["filter"] = filter_params
        
        try:
            result = await self.session.call_tool(tool_name, arguments)
            if result and result.content and len(result.content) > 0:
                return result.content[0].text
            else:
                return f"No results returned from {tool_name}"
        except Exception as e:
            return f"Error calling tool {tool_name}: {str(e)}"

    async def chat_loop(self):
        """Run an interactive search loop"""
        print("\nQdrant Search Client Started!")
        print("Available commands:")
        print("  search <collection> <query> [top_k] [score_threshold]")
        print("  list               - List available collections")
        print("  quit               - Exit the client")
        
        while True:
            try:
                command = input("\nCommand: ").strip()
                
                if command.lower() == 'quit':
                    break
                
                if command.lower() == 'list':
                    response = await self.session.list_tools()
                    tools = response.tools
                    print("\nAvailable collections:")
                    for tool in tools:
                        # Handle both collections-list and search-collection tools
                        if tool.name == "collections-list":
                            continue  # Skip the list tool itself
                        elif tool.name.startswith("search-"):
                            collection_name = tool.name[len("search-"):]
                            print(f"- {collection_name}: {tool.description}")
                        else:
                            print(f"- {tool.name}: {tool.description}")
                    continue
                    
                if command.lower().startswith('search '):
                    parts = command.split(' ', 3)  # Split into at most 4 parts
                    
                    if len(parts) < 3:
                        print("Usage: search <collection> <query> [top_k] [score_threshold]")
                        continue
                    
                    collection = parts[1]
                    query = parts[2]
                    top_k = 5
                    score_threshold = 0.4
                    
                    if len(parts) >= 4:
                        additional_params = parts[3].split()
                        if len(additional_params) >= 1:
                            try:
                                top_k = int(additional_params[0])
                            except ValueError:
                                print("Invalid top_k value. Using default (5).")
                        
                        if len(additional_params) >= 2:
                            try:
                                score_threshold = float(additional_params[1])
                            except ValueError:
                                print("Invalid score_threshold value. Using default (0.4).")
                    
                    print(f"\nSearching '{collection}' for '{query}' (top_k={top_k}, threshold={score_threshold})...")
                    result = await self.search(collection, query, top_k, score_threshold)
                    print("\n" + result)
                    continue
                
                print("Unknown command. Type 'list' to see available collections or 'quit' to exit.")
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python search_client_mcp.py <path_to_server_script>")
        server_path = os.path.dirname(__file__) + "/search_mcp.py"
    else:
        server_path = sys.argv[1]
            
    client = QdrantSearchClient()
    try:
        await client.connect_to_server(server_path)
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
