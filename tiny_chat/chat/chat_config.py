import json
import os


DEFAULT_CHAT_CONFIG_PATH = "chat_app_config.json"


class ChatConfig:
    """
    チャットアプリケーションの設定を管理するクラス
    """

    def __init__(
        self,
        server_url: str = "http://localhost:11434/v1",
        api_key: str = "dummy-key",
        selected_model: str = "hf.co/SakanaAI/TinySwallow-1.5B-Instruct-GGUF",
        meta_prompt: str = "",
        message_length: int = 8000,
        max_completion_tokens: int = 1000,
        context_length: int = 2000,
        uri_processing: bool = True,
        is_azure: bool = False,
        session_only_mode: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        rag_process_prompt: str = "関連文書が有効な場合は回答に役立ててください。\n関連文書:\n",
        use_hyde: bool = False,
        use_step_back: bool = False,
        use_web: bool = True,
        web_top_k: int = 3,
        use_multi: bool = False,
        use_deep: bool = False,
        timeout: float = 30.,
        **kwargs
    ):
        self.server_url = server_url
        self.api_key = api_key
        self.selected_model = selected_model
        self.meta_prompt = meta_prompt
        self.message_length = message_length
        self.max_completion_tokens = max_completion_tokens
        self.context_length = context_length
        self.uri_processing = uri_processing
        self.is_azure = is_azure
        self.session_only_mode = session_only_mode
        self.temperature = temperature
        self.top_p = top_p
        self.rag_process_prompt = rag_process_prompt
        self.use_hyde = use_hyde
        self.use_step_back = use_step_back
        self.use_web = use_web
        self.web_top_k = web_top_k
        self.use_multi = use_multi
        self.use_deep = use_deep
        self.timeout = timeout

    @classmethod
    def load(cls, file_path: str) -> 'ChatConfig':
        """
        設定ファイルから設定を読み込む

        Args:
            file_path (str): 設定ファイルのパス

        Returns:
            Config: 設定オブジェクト
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                return cls(**config_data)
            else:
                return cls()
        except Exception:
            return cls()

    def save(self, file_path: str) -> bool:
        """
        設定をファイルに保存する

        Args:
            file_path (str): 設定ファイルのパス

        Returns:
            bool: 保存が成功したかどうか
        """
        # セッションのみモードが有効な場合はファイル保存しない
        if self.session_only_mode:
            return True
            
        try:
            config_data = {
                'server_url': self.server_url,
                'api_key': self.api_key,
                'selected_model': self.selected_model,
                'meta_prompt': self.meta_prompt,
                'message_length': self.message_length,
                'max_completion_tokens': self.max_completion_tokens,
                'context_length': self.context_length,
                'uri_processing': self.uri_processing,
                'is_azure': self.is_azure,
                'session_only_mode': self.session_only_mode,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'rag_process_prompt': self.rag_process_prompt,
                'use_hyde': self.use_hyde,
                'use_step_back': self.use_step_back,
                'use_web': self.use_web,
                'web_top_k': self.web_top_k,
                'use_multi': self.use_multi,
                "use_deep": self.use_deep,
                "timeout": self.timeout
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False


class ModelManager:
    """モデル関連の機能を提供するクラス"""

    @staticmethod
    def fetch_available_models(
            server_url, api_key, openai_client=None, is_azure=False):
        """
        利用可能なモデル一覧を取得

        Args:
            server_url (str): サーバーURL
            api_key (str): APIキー
            openai_client: OpenAIクライアント（オプション）

        Returns:
            tuple: (モデルリスト, 成功フラグ)
        """
        if is_azure:
            # deployments APIで代替可だがあまり使わない？
            return [], True

        try:
            if openai_client is None:
                from openai import OpenAI

                openai_client = OpenAI(
                    base_url=server_url,
                    api_key=api_key
                )

            # モデル一覧を取得
            models = openai_client.models.list()
            if models.data is None:
                return [], True
            model_ids = [model.id for model in models.data]

            if model_ids:
                return model_ids, True  # 成功を示すフラグを返す

            return [], False  # 失敗を示すフラグを返す
        except Exception as e:
            print(f"モデル取得エラー: {str(e)}")
            return [], False  # 失敗を示すフラグを返す

    @staticmethod
    def update_models_on_server_change(new_server_url, api_key, current_model, is_azure=False):
        """
        サーバーURLが変更された時にモデルリストを更新

        Args:
            new_server_url (str): 新しいサーバーURL
            api_key (str): APIキー
            current_model (str): 現在選択中のモデル

        Returns:
            tuple: (新しいモデルリスト, 選択されるべきモデル, 成功フラグ)
        """
        # 新しいモデルリストを取得
        new_models, success = ModelManager.fetch_available_models(
            new_server_url, api_key, is_azure=is_azure)

        # 現在のモデルが新しいリストに含まれていない場合は先頭のモデルを選択
        if current_model not in new_models and new_models:
            return new_models, new_models[0], success

        # 現在のモデルが含まれている場合はそのまま
        return new_models, current_model, success