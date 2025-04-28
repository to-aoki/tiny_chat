import os
import sys
from typing import Dict, Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAIError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tiny_chat.api.api_util import get_qdrant_manager, get_llm_api, get_collections, search

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
        search_result = search(query, manager=qdrant_manager, collection_info=collection, chat_config=chat_config)
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
