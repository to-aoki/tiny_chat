import os
import sys

from pydantic import BaseModel
from fastapi import FastAPI
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from tiny_chat.database.components.search import search_documents
from tiny_chat.api.api_util import get_qdrant_manager, get_collections
from tiny_chat.database.qdrant.rag_strategy import RagStrategyFactory


qdrant_manager = get_qdrant_manager()
available_collections = get_collections(qdrant_manager)


app = FastAPI(
    title="TinyChat DataBase REST API",
    version="0.0.1",
    description="TinyChat DataBase REST API",
)

class QueryRequest(BaseModel):
    query: str
    collection_name: str


@app.post("/retrieve")
async def search(query_request: QueryRequest):
    query = query_request.query
    collection = query_request.collection_name
    if query is None or query == "" or collection is None or collection == "":
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
                page = result.payload.get('page', '')
                context_items.append(
                    {
                        "source": source,
                        "content": text,
                        "page": page
                    }
                )
    except Exception as e:
        print(f"Error in search: {e}")
        return []

    return context_items


class IndexRequest(BaseModel):
    collection_name: str
    source: str
    text: str
    page: Optional[int] = -1


@app.post("/create")
async def create(index_request: IndexRequest):
    source = index_request.source
    text = index_request.text
    collection = index_request.collection_name
    page = index_request.page if index_request.page is not None else -1
    if source is None or source == "" or text is None or text == "" or collection is None or collection == "":
        return {"create": False}

    collection_info = available_collections.get(collection, None)
    if collection_info is None:
        return {"create": False}

    collection_name = collection_info.collection_name
    try:
        strategy = RagStrategyFactory.get_strategy(collection_info.rag_strategy, collection_info.use_gpu)
        metadata = {"source" : source}
        if page > -1:
            metadata["page"] = page
        qdrant_manager.add_document(
            text, metadata, collection_name,
            strategy=strategy, chunk_size=collection_info.chunk_size, chunk_overlap=collection_info.chunk_overlap)
    except Exception as e:
        print(f"Error in create: {e}")
        return {"create": False}

    return {"create": True}


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="TinyChat DataBase REST API")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (for remote mode)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8082,
        help="Port to bind the server to (for remote mode)"
    )
    return parser.parse_args()


def main():
    import uvicorn
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
