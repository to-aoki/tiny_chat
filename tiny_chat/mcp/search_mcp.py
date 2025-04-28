import os
import sys
import argparse
import threading
from typing import Dict, Any, Optional

from mcp.server.fastmcp import FastMCP, Context
from contextlib import asynccontextmanager

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tiny_chat.database.qdrant.qdrant_manager import QdrantManager
from tiny_chat.database.qdrant.collection import Collection
from tiny_chat.database.qdrant.rag_strategy import RagStrategyFactory
from tiny_chat.database.database_config import DatabaseConfig, DEFAULT_CONFIG_PATH


# シングルトンとしてQdrantManagerを管理するグローバルインスタンス
_qdrant_manager_instance: Optional[QdrantManager] = None
# スレッドセーフな操作のためのロックオブジェクト
_qdrant_manager_lock = threading.RLock()
# 利用可能なコレクションリスト
available_collections = {}


def get_qdrant_manager() -> QdrantManager:
    """
    QdrantManagerのシングルトンインスタンスを取得する関数
    
    既にインスタンスが初期化されている場合はそれを返し、
    まだ初期化されていない場合は新しいインスタンスを作成して返します。
    排他ロックを使用してマルチスレッド環境でも安全に動作します。
    
    Returns:
        QdrantManager: シングルトンインスタンス
    """
    global _qdrant_manager_instance
    
    # Double-checked locking pattern
    # 最初のチェックはロック外で行うことで、インスタンスが既にある場合のパフォーマンスを向上
    if _qdrant_manager_instance is None:
        # ロックを取得してから二重確認
        with _qdrant_manager_lock:
            # 2回目のチェック - 他のスレッドが初期化を完了していないか確認
            if _qdrant_manager_instance is None:
                # 初期化がまだ行われていない場合は設定を読み込んでインスタンスを作成
                config_file_path = os.environ.get("DB_CONFIG", DEFAULT_CONFIG_PATH)
                db_config = DatabaseConfig.load(config_file_path)
                _qdrant_manager_instance = QdrantManager(**db_config.__dict__)
    
    return _qdrant_manager_instance


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
async def lifespan_manager(fastmcp_app: FastMCP):
    """Lifespan handler for FastMCP - setup that runs on app startup"""
    # Register search tools for all collections
    qdrant_mgr = get_qdrant_manager()
    await register_search_tools(fastmcp_app, qdrant_mgr)
    yield {}  # This is required for the lifespan context manager


def get_collection_description(collection_name: str) -> str:
    """
    Get the description of a collection from the collection_descriptions collection
    
    Args:
        collection_name: Name of the collection
        
    Returns:
        str: Description of the collection, or default description if not found
    """
    qdrant_mgr = get_qdrant_manager()
    collection_info = Collection.load(collection_name, qdrant_mgr)
    
    if collection_info:
        return collection_info
    else:
        # 説明情報なし
        return f"Search {collection_name} collection"


async def register_search_tools(app, qdrant_mgr):
    """Dynamically register search tools for each collection"""
    collections = qdrant_mgr.get_collections()
    
    # Skip the collection_descriptions collection
    collections = [c for c in collections if c != Collection.STORED_COLLECTION_NAME]
    
    for collection_name in collections:
        collection_info = get_collection_description(collection_name)
        if isinstance(collection_info, str):
            continue

        if not collection_info.show_mcp:
            continue

        available_collections[collection_name] = collection_info

        # Create a local function factory that binds the current collection_name
        def create_search_tool(name, collection):
            # Define a tool for this collection
            tool_description = getattr(collection, 'description', f"Search {name} collection")
            
            @app.tool(
                name=f"search-{name}",
                description=tool_description
            )
            async def search_collection_tool(
                query: str,
                top_k: int = getattr(collection, 'top_k', 3),
                score_threshold: float = getattr(collection, 'score_threshold', 0.4),
                ctx: Context = None
            ) -> str:
                """Search the specified collection"""
                if ctx:
                    await ctx.info(f"Searching collection '{name}' for: {query}")
                    
                return await search_collection(name, {
                    "query": query,
                    "top_k": top_k,
                    "score_threshold": score_threshold
                })
            
            # Each tool needs a unique name in the module scope
            search_collection_tool.__name__ = f"search_{name}_tool"
            
            return search_collection_tool
        
        # Create and register the tool for this collection
        create_search_tool(collection_name, collection_info)


async def search_collection(
    collection_name: str, 
    arguments: Dict[str, Any]
) -> str:
    """
    Search a collection with the given query and parameters
    
    Args:
        collection_name: Name of the collection to search
        arguments: Search parameters including query, top_k, etc.
        
    Returns:
        str: Search results formatted as text
    """
    qdrant_mgr = get_qdrant_manager()
    query = arguments.get("query", "")
    collection = available_collections.get(collection_name)
    if collection is None:
        return f"Error searching collection '{collection_name}': collection not found."
    top_k = arguments.get("top_k", getattr(collection, 'top_k', 3))
    score_threshold = arguments.get("score_threshold", getattr(collection, 'score_threshold', 0.4))
    rag_strategy = getattr(collection, 'rag_strategy', None)
    if rag_strategy is None:
        return f"Error searching collection '{collection_name}': rag_strategy not found."
    use_gpu = getattr(collection, 'use_gpu', False)

    try:
        results = qdrant_mgr.query_points(
            query=query,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
            strategy=RagStrategyFactory.get_strategy(rag_strategy, use_gpu)
        )

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
    args = parse_args()
    
    # シングルトンインスタンスを事前に初期化して取得
    get_qdrant_manager()
    
    # Create FastMCP instance
    app = FastMCP(
        name="tiny-chat-search-mcp",
        instructions="Search through vector database collections",
        host=args.host,       # Listen on specified host
        port=args.port,       # Port for the server
        debug=args.debug,     # Debug mode based on args
        lifespan=lifespan_manager  # Assign lifespan function
    )

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
