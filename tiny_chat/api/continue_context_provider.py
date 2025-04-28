"""
    {
      "name": "http",
      "params": {
        "url": "http://localhost:8081/retrieve",
        "title": "http",
        "description": "Continue HTTP Context Provider",
        "displayTitle": "qdrant",
        "options": {
          "collection": "default"
        }
      }
    }
"""

import os
import sys

from pydantic import BaseModel
from fastapi import FastAPI
from typing import Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tiny_chat.database.components.search import search_documents
from tiny_chat.api.api_util import get_qdrant_manager, get_collections


qdrant_manager = get_qdrant_manager()
available_collections = get_collections(qdrant_manager)


class ContextProviderInput(BaseModel):
    query: str
    fullInput: str
    options: Dict[str, Any]


app = FastAPI(
    title="Continue Context Provider",
    version="0.0.1",
    description="A Continue Context Provider, with Qdrant Search.",
)


@app.post("/retrieve")
async def tiny_search(context_provider_input: ContextProviderInput):
    query = context_provider_input.fullInput
    if query is None or query == "":
        return []
    collection = context_provider_input.options.get("collection", None)
    if collection is None:
        return []
    collection_info = available_collections.get(collection, None)
    if collection_info is None:
        return []

    collection_name = collection_info.collection_name
    top_k = collection_info.top_k
    score_threshold = collection_info.score_threshold

    context_items = []
    try:
        results = search_documents(
            query=query,
            qdrant_manager=qdrant_manager,
            collection_name=collection_name,
            top_k=top_k,
            score_threshold=score_threshold,
        )
        if results:
            for result in results:
                source = result.payload.get('source', '')
                text = result.payload.get('text', '')
                context_items.append(
                    {
                        "name": source,
                        "description": source,
                        "content": text,
                    }
                )
    except:
        return []

    return context_items


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="TinyChat Continue Context Provider")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (for remote mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8081,
        help="Port to bind the server to (for remote mode)"
    )
    return parser.parse_args()


def main():
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
