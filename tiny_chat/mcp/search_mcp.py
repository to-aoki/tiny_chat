import os
import sys
import argparse
from typing import Dict, Any

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from tiny_chat.database.qdrant.qdrant_manager import QdrantManager
from tiny_chat.database.qdrant.collection import Collection
from tiny_chat.database.database_config import DatabaseConfig, DEFAULT_CONFIG_PATH


# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="TinyChat Search MCP Server")
    parser.add_argument(
        "--mode", 
        choices=["local", "remote"], 
        default="local",
        help="Server mode: 'local' for stdio or 'remote' for external access"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Host to bind the server to (for remote mode)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to (for remote mode)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()

# Define lifespan context manager
@asynccontextmanager
async def lifespan_manager(fastmcp_app: FastMCP, qdrant_mgr):
    """Lifespan handler for FastMCP - setup that runs on app startup"""
    # Register search tools for all collections
    await register_search_tools(fastmcp_app, qdrant_mgr)
    yield {}  # This is required for the lifespan context manager

# FastMCP instance will be created in main()

# Global variables
available_collections = {}


def get_collection_description(collection_name: str, qdrant_mgr) -> str:
    """
    Get the description of a collection from the collection_descriptions collection
    
    Args:
        collection_name: Name of the collection
        qdrant_mgr: Instance of QdrantManager
        
    Returns:
        str: Description of the collection, or default description if not found
    """
    # Ensure the collection_descriptions collection exists
    Collection.ensure_collection_descriptions_exists(qdrant_mgr)
    
    # Try to load the collection information
    collection_info = Collection.load(collection_name, qdrant_mgr)
    
    if collection_info:
        return collection_info
    else:
        return f"Search {collection_name} collection"


# Function definition for FastMCP tool registration in main()
async def list_collections(ctx: Context, qdrant_mgr) -> Dict[str, str]:
    """List available collections for searching"""
    # Get all available collections
    collections = qdrant_mgr.get_collections()
    
    # Skip the collection_descriptions collection
    collections = [c for c in collections if c != Collection.STORED_COLLECTION_NAME]
    
    result = {}
    for collection_name in collections:
        collection_info = get_collection_description(collection_name, qdrant_mgr)
        
        # Handle different return types from get_collection_description
        if isinstance(collection_info, str):
            description = collection_info
        else:
            description = getattr(collection_info, 'description', f"Search {collection_name} collection")
            
        result[collection_name] = description
        
    return result


async def register_search_tools(app, qdrant_mgr):
    """Dynamically register search tools for each collection"""
    collections = qdrant_mgr.get_collections()
    
    # Skip the collection_descriptions collection
    collections = [c for c in collections if c != Collection.STORED_COLLECTION_NAME]
    
    for collection_name in collections:
        collection_info = get_collection_description(collection_name, qdrant_mgr)
        
        # Ensure collection_info is properly handled regardless of its type
        if isinstance(collection_info, str):
            # If get_collection_description returned a string
            collection = type('Collection', (), {
                'description': collection_info,
                'top_k': 3,
                'score_threshold': 0.4
            })
        else:
            # If get_collection_description returned an object
            collection = collection_info
            
        available_collections[collection_name] = collection
        
        # Create a local function factory that binds the current collection_name
        def create_search_tool(coll_name, coll):
            # Define a tool for this collection
            tool_description = getattr(coll, 'description', f"Search {coll_name} collection")
            
            @app.tool(
                name=f"search-{coll_name}",  # Use hyphen instead of underscore
                description=tool_description
            )
            async def search_collection_tool(
                query: str,
                top_k: int = getattr(coll, 'top_k', 3),
                score_threshold: float = getattr(coll, 'score_threshold', 0.4),
                ctx: Context = None
            ) -> str:
                """Search the specified collection"""
                if ctx:
                    await ctx.info(f"Searching collection '{coll_name}' for: {query}")
                    
                return await search_collection(coll_name, {
                    "query": query,
                    "top_k": top_k,
                    "score_threshold": score_threshold
                }, qdrant_mgr)
            
            # Each tool needs a unique name in the module scope
            search_collection_tool.__name__ = f"search_{coll_name}_tool"
            
            return search_collection_tool
        
        # Create and register the tool for this collection
        create_search_tool(collection_name, collection)


async def search_collection(
    collection_name: str, 
    arguments: Dict[str, Any],
    qdrant_mgr
) -> str:
    """
    Search a collection with the given query and parameters
    
    Args:
        collection_name: Name of the collection to search
        arguments: Search parameters including query, top_k, etc.
        qdrant_mgr: Instance of QdrantManager
        
    Returns:
        str: Search results formatted as text
    """
    query = arguments.get("query", "")
    collection = available_collections.get(collection_name)
    top_k = arguments.get("top_k", getattr(collection, 'top_k', 3))
    score_threshold = arguments.get("score_threshold", getattr(collection, 'score_threshold', 0.4))

    try:
        settings = {
            'collection_name': collection_name,
            'description': getattr(collection, 'description', ''),
            'rag_strategy': getattr(collection, 'rag_strategy', 'bm25_static'),
            'use_gpu': getattr(collection, 'use_gpu', False),
            'chunk_size': getattr(collection, 'chunk_size', 1024),
            'chunk_overlap': getattr(collection, 'chunk_overlap', 24)
        }
        qdrant_mgr.update_settings(**settings)
        # Execute search query
        results = qdrant_mgr.query_points(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
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
            return f"No results found for query: '{query}' in collection '{collection_name}'"
        
        # Return results as a string
        return f"Search results for '{query}' in collection '{collection_name}':\n\n" + \
               "\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching collection '{collection_name}': {str(e)}"


def main():
    """
    Main function to start the TinyChat Search MCP Server
    """
    # Parse command line arguments
    args = parse_args()
    
    # Initialize database config and Qdrant manager
    config_file_path = os.environ.get("DB_CONFIG", DEFAULT_CONFIG_PATH)
    db_config = DatabaseConfig.load(config_file_path)
    qdrant_mgr = QdrantManager(**db_config.__dict__)
    
    # Create a custom lifespan function that includes qdrant_mgr
    @asynccontextmanager
    async def app_lifespan(fastmcp_app):
        # FastMCP expects a direct async generator as lifespan
        # Register search tools for all collections
        await register_search_tools(fastmcp_app, qdrant_mgr)
        yield {}
    
    # Create FastMCP instance
    app = FastMCP(
        name="tiny-chat-search-mcp",
        instructions="Search through vector database collections",
        host=args.host,       # Listen on specified host
        port=args.port,       # Port for the server
        debug=args.debug,     # Debug mode based on args
        lifespan=app_lifespan  # Assign lifespan function
    )
    
    # Register collections-list tool
    @app.tool(name="collections-list", description="List available Qdrant collections")
    async def collections_list_tool(ctx: Context) -> Dict[str, str]:
        return await list_collections(ctx, qdrant_mgr)
    
    # Set binary mode for stdin/stdout on Windows if running in local mode
    if args.mode == "local" and sys.platform == 'win32':
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    
    # Run FastMCP with either stdio or SSE transport based on mode
    transport = "stdio" if args.mode == "local" else "sse"
    print(f"Starting TinyChat Search MCP Server in {args.mode} mode")
    if args.mode == "remote":
        print(f"Server running at http://{args.host}:{args.port}")
    
    app.run(transport=transport)


if __name__ == "__main__":
    main()
