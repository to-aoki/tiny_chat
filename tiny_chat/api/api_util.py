import os

from openai import AzureOpenAI, OpenAI
from typing import Dict, Any, Union

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


def search(
    query: str, manager: QdrantManager, collection_info: Collection, chat_config: ChatConfig,
    query_processer=None
) -> Dict[str, Any]:
    collection_name = collection_info.collection_name
    top_k = collection_info.top_k
    score_threshold = collection_info.score_threshold

    try:
        dense_text = None
        multi_queries = []
        if query_processer is not None:
            modify_queries = query_processer.transform(query)
            if isinstance(modify_queries, list):
                multi_queries = modify_queries
            else:
                dense_text = modify_queries

        if len(multi_queries) > 0:
            from tiny_chat.utils.query_preprocessor import QueryPlanner
            full_result = []
            for q in multi_queries:
                search_result = search_documents(
                    query=q.query,
                    qdrant_manager=manager,
                    collection_name=collection_name,
                    top_k=top_k,
                    score_threshold=score_threshold,
                )
                full_result.append(
                    search_result
                )
            results = QueryPlanner.result_merge(full_result)
            results = results[:top_k]
        else:
            results = search_documents(
                query=query,
                qdrant_manager=manager,
                collection_name=collection_name,
                top_k=top_k,
                score_threshold=score_threshold,
                dense_text=dense_text
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
