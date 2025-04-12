import anyio
import os
import sys
from typing import Dict, Any
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tiny_chat.database.qdrant.qdrant_manager import QdrantManager
from tiny_chat.database.qdrant.collection import Collection
from tiny_chat.database.database_config import DatabaseConfig, DEFAULT_CONFIG_PATH

server = Server("tiny-chat-search-mcp")
config_file_path = os.environ.get("DB_CONFIG", DEFAULT_CONFIG_PATH)

# Initialize database config and Qdrant manager
db_config = DatabaseConfig.load(config_file_path)
qdrant_manager = QdrantManager(
    **db_config.__dict__
)


def get_collection_description(collection_name: str) -> str:
    """
    Get the description of a collection from the collection_descriptions collection
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        str: Description of the collection, or default description if not found
    """
    # Ensure the collection_descriptions collection exists
    Collection.ensure_collection_descriptions_exists(qdrant_manager)
    
    # Try to load the collection information
    collection_info = Collection.load(collection_name, qdrant_manager)
    
    if collection_info:
        return collection_info.description
    else:
        return f"Search {collection_name} collection"


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available Qdrant search tools"""
    tools = []
    
    # Get all available collections
    collections = qdrant_manager.get_collections()
    
    # Skip the collection_descriptions collection
    collections = [c for c in collections if c != Collection.STORED_COLLECTION_NAME]
    
    # Create a search tool for each collection
    for collection_name in collections:
        description = get_collection_description(collection_name)
        
        tools.append(
            types.Tool(
                name=f"search_{collection_name}",
                description=description,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5
                        },
                        "score_threshold": {
                            "type": "number",
                            "description": "Minimum score threshold for results",
                            "default": 0.4
                        },
                        "filter": {
                            "type": "object",
                            "description": "Optional filter parameters"
                        }
                    },
                    "required": ["query"]
                }
            )
        )
    
    return tools


@server.call_tool()
async def handle_call_tool(
    name: str,
    arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool execution"""
    if not arguments:
        raise ValueError("Missing arguments")

    # Check if the tool name starts with "search_"
    if name.startswith("search_"):
        collection_name = name[len("search_"):]
        return await search_collection(collection_name, arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")

async def search_collection(
    collection_name: str, 
    arguments: Dict[str, Any]
) -> list[types.TextContent]:
    """
    Search a collection with the given query and parameters
    
    Args:
        collection_name: Name of the collection to search
        arguments: Search parameters including query, top_k, etc.
        
    Returns:
        list[types.TextContent]: Search results formatted as text
    """
    query = arguments.get("query", "")
    top_k = arguments.get("top_k", 5)
    score_threshold = arguments.get("score_threshold", 0.4)
    filter_params = arguments.get("filter", None)
    
    try:
        # Execute search query
        results = qdrant_manager.query_points(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_params=filter_params
        )
        
        # Format the results
        formatted_results = []
        for i, result in enumerate(results):
            text = result.payload.get("text", "")
            score = result.score
            
            # Filter out metadata fields that shouldn't be displayed
            metadata = {k: v for k, v in result.payload.items() 
                       if k not in ["text", "chunk_index", "chunk_total"]}
            
            formatted_results.append(f"Result {i+1} [Score: {score:.4f}]\n{text}\nMetadata: {metadata}")
        
        # If no results found
        if not formatted_results:
            return [
                types.TextContent(
                    type="text",
                    text=f"No results found for query: '{query}' in collection '{collection_name}'"
                )
            ]
        
        # Return results
        return [
            types.TextContent(
                type="text",
                text=f"Search results for '{query}' in collection '{collection_name}':\n\n" + 
                     "\n\n".join(formatted_results)
            )
        ]
        
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"Error searching collection '{collection_name}': {str(e)}"
            )
        ]


def main() -> int:
    """Run the server"""
    # Set binary mode for stdin/stdout on Windows
    if sys.platform == 'win32':
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    async def arun():
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="tiny-chat-search-mcp",
                    server_version="0.1.1",
                    capabilities=server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                )
            )
    anyio.run(arun)
    return 0


if __name__ == "__main__":
    anyio.run(main())
