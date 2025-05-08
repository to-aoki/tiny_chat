from pydantic import BaseModel
from typing import List, Dict, Any
from duckduckgo_search import DDGS

class DDGSResult(BaseModel):
    payload: Dict[str, Any]


def search_web(query, max_results=3, region='jp-ja'):
    formatted_results = []
    try:
        with DDGS() as ddgs:
            ddgs_results = ddgs.text(query, max_results=max_results, region=region)
            for r in ddgs_results:
                formatted_results.append(
                    DDGSResult(
                        payload={
                            "text": r.get('body', 'No Content'),
                            "source": r.get('href', '#')
                        }
                    )
                )
            return formatted_results
    except Exception as e:
        return formatted_results