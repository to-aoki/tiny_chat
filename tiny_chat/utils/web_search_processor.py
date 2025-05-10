from pydantic import BaseModel
from typing import Dict, Any
from duckduckgo_search import DDGS

class DDGSResult(BaseModel):
    payload: Dict[str, Any]


def search_web(query, max_results=3, region='jp-ja', logger=None):
    formatted_results = []
    try:
        with DDGS() as ddgs:
            ddgs_results = ddgs.text(query, max_results=max_results, region=region)
            for r in ddgs_results:
                formatted_results.append(
                    DDGSResult(
                        payload={
                            "text": r.get('body', 'No Content'),
                            "source": r.get('href', '#'),
                            "page": ""
                        }
                    )
                )
            return formatted_results
    except:
        if logger:
            import traceback
            error_message = traceback.format_exc()
            logger.error(f"search_webが失敗しました: {str(error_message)}")
        return formatted_results