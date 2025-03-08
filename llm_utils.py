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
