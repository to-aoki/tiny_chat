from urllib.parse import urlparse, urlunparse

import requests
from pydantic import BaseModel

def get_llm_client(server_url, api_key='dummy-key', is_azure=False, azure_api_version="2024-06-01"):
    if is_azure:
        from openai import AzureOpenAI

        return AzureOpenAI(
            api_key=api_key,
            azure_endpoint=server_url,
            api_version=azure_api_version
        )
    else:
        from openai import OpenAI

        return OpenAI(
            base_url=server_url,
            api_key=api_key
        )

def reset_ollama_model(server_url="http://localhost:11434/v1", model="llama3"):

    try:
        from urllib.parse import urlparse, urljoin
        import requests

        parsed_original_url = urlparse(server_url)
        if not parsed_original_url.scheme or not parsed_original_url.netloc:
            raise ValueError("無効なベースURLです。スキーム (http/https) とホスト名（:ポート）を含めてください。")

        # 例: "http://localhost:11434/v1" -> "http://localhost:11434"
        ollama_server_root = f"{parsed_original_url.scheme}://{parsed_original_url.netloc}"

        target_url = urljoin(ollama_server_root, "/api/generate")

        payload = {
            "model": model,
            "keep_alive": 0,
        }
        response = requests.post(target_url, json=payload)
        response.raise_for_status()
        return True

    except:
        # ollamaでないサーバーの場合は単にエラーとなるため詳細補足しない
        return False


def convert_openai_response_format(pydantic_instance:BaseModel):
    # response_format用の表現に変える
    return {
        "type": "json_schema",
        "json_schema": {
            "schema": pydantic_instance.model_json_schema(),
            "name": pydantic_instance.__name__,
            "strict": True,
        }
    }

def identify_server(base_url: str, timeout: int = 3) -> str:
    parsed_original_url = urlparse(base_url.rstrip('/'))
    root_url_parts = (parsed_original_url.scheme, parsed_original_url.netloc,  '', '', '', '')
    root_url = urlunparse(root_url_parts).rstrip('/')

    try:
        # 1. Ollama Check
        ollama_root_check_url = f"{root_url}/"
        response = requests.get(ollama_root_check_url, timeout=timeout)
        if response.status_code == 200 and "Ollama is running" in response.text:
            return "ollama"
    except requests.exceptions.RequestException:
        pass

    try:
        # 2. vLLM Health Check
        vllm_health_url = f"{root_url}/health"
        response = requests.get(vllm_health_url, timeout=timeout)
        if response.status_code == 200:
            return "vllm"
    except requests.exceptions.RequestException:
        pass

    return "other"
