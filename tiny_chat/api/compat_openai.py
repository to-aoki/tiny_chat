import os
import sys

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAIError
from openai import AzureOpenAI, OpenAI
from typing import Dict, Any, AsyncGenerator, Union

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tiny_chat.utils.llm_utils import get_llm_client
from tiny_chat.chat.chat_config import ChatConfig, DEFAULT_CHAT_CONFIG_PATH
from tiny_chat.database.qdrant.qdrant_manager import QdrantManager
from tiny_chat.database.components.search import search_documents
from tiny_chat.database.qdrant.collection import Collection
from tiny_chat.database.database_config import DatabaseConfig, DEFAULT_CONFIG_PATH


def get_qdrant_manager() -> QdrantManager:
    config_file_path = os.environ.get("DB_CONFIG", DEFAULT_CONFIG_PATH)
    db_config = DatabaseConfig.load(config_file_path)
    return QdrantManager(**db_config.__dict__)


def get_llm_api() -> Union[AzureOpenAI | OpenAI]:
    config_file_path = os.environ.get("LLM_CONFIG", DEFAULT_CHAT_CONFIG_PATH)
    chat_config = ChatConfig.load(config_file_path)
    llm_api = get_llm_client(
        server_url=chat_config.server_url,
        api_key=chat_config.api_key,
        is_azure=chat_config.is_azure
    )
    return llm_api, chat_config


def get_collections(manager: QdrantManager) -> Dict[str, Collection]:
    collections_dict = {}
    collections = manager.get_collections()
    collections = [c for c in collections if c != Collection.STORED_COLLECTION_NAME]
    for collection_name in collections:
        collection_info = Collection.load(collection_name, manager)
        if isinstance(collection_info, str):
            continue
        if not collection_info.show_mcp:
            continue
        collections_dict[collection_name] = collection_info
    return collections_dict


def search(query: str, manager: QdrantManager, collection_info: Collection, config: ChatConfig) -> Dict[str, Any]:
    collection_name = collection_info.collection_name
    top_k = collection_info.top_k
    score_threshold = collection_info.score_threshold

    try:
        results = search_documents(
            query=query,
            qdrant_manager=manager,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        search_context = ""
        if results:
            search_context = chat_config.rag_process_prompt

            for result in results:
                source = result.payload.get('source', '')
                text = result.payload.get('text', '')[:chat_config.context_length]
                search_context += f"{source}:\n{text}\n\n"

        return search_context

    except Exception as e:
        return f"Error searching collection '{collection_name}': {str(e)}"


qdrant_manager = get_qdrant_manager()
llm_api, chat_config = get_llm_api()
available_collections = get_collections(qdrant_manager)


app = FastAPI(
    title="OpenAI Compatible API Proxy",
    version="0.0.1",
    description="A proxy API compatible with OpenAI Chat API v1, with Qdrant RAG.",
)


@app.post(
    "/v1/chat/completions",
    summary="Chat interface using RAG",
    description="Receives OpenAI chat completion requests, searches the database and appends other information, "
                "then forwards them to the configured API endpoint.",
)
async def chat_completions_proxy(
    request: Request,
    request_body: Dict[str, Any] = Body(...)
):
    """
    Proxies requests to the configured Chat Completions API endpoint.
    """
    is_streaming = request_body.get("stream", False)
    model = request_body.get("model", None)
    collection = None
    if model is not None:
        collection = available_collections.get(model, None)

    if collection is not None and request_body["messages"] and request_body["messages"][-1].get("role") == "user":
        query = request_body["messages"][-1]["content"]
        search_result = search(query, manager=qdrant_manager, collection_info=collection, config=chat_config)
        request_body["messages"][-1]["content"] += search_result

    request_body["model"] = chat_config.selected_model
    create_params = request_body
    try:
        if is_streaming:
            async def stream_generator() -> AsyncGenerator[str, None]:
                response = llm_api.chat.completions.create(**create_params)
                for chunk in response:
                    content = chunk.model_dump_json()
                    yield f"data: {content}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(stream_generator(), media_type="text/event-stream")

        else:
            completion = llm_api.chat.completions.create(**create_params)
            response_data = completion.model_dump(exclude_unset=True)
            return JSONResponse(content=response_data)

    except OpenAIError as e:
        detail = e.body or {"error": {"message": str(e), "type": type(e).__name__, "code": getattr(e, 'code', None)}}
        raise HTTPException(
            status_code=getattr(e, 'status_code', 500),
            detail=detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": "An internal server error occurred.", "type": type(e).__name__}}
        )


@app.get(
    "/v1/models",
    summary="Chat Completions Proxy",
    description="Receives OpenAI chat completion requests and forwards them to the configured API endpoint.",
)
async def rag_collection_models(
    request: Request
):
    available_collections = get_collections(qdrant_manager)
    data = [{"id": collection_name, "object": "model"} for collection_name in available_collections.keys()]
    return JSONResponse({
          "object": "list",
          "data": data
        }
    )


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="TinyChat RAG OpenAI Chat Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (for remote mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind the server to (for remote mode)"
    )
    return parser.parse_args()


def main():
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
