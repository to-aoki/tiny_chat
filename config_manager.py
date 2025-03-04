import json
import os
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """アプリケーション設定を保持するデータクラス"""
    server_url: str = "http://127.0.0.1:11434/v1/"
    api_key: str = "dummy-key"
    selected_model: str = "hf.co/SakanaAI/TinySwallow-1.5B-Instruct-GGUF:Q5_K_M"
    meta_prompt: str = ""
    context_length: int = 8000  # 添付ファイルの長さ制限
    message_length: int = 16000  # メッセージ全体の長さ制限
    uri_processing: bool = False  # メッセージ中のURLを解決してダウンロードするか
    is_azure: bool = False  # Azure OpenAIを利用するか

    @classmethod
    def load(cls, config_file):
        """
        設定ファイルから設定を読み込む
        ファイルが存在しない場合はデフォルト設定を返す

        Args:
            config_file (str): 設定ファイルのパス

        Returns:
            Config: 読み込んだ設定
        """
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_dict = json.load(f)
                    # 辞書から Config オブジェクトを作成
                    return cls(**config_dict)
            except Exception as e:
                pass
        return Config()

    def save(self, config_file):
        """
        Config オブジェクトをファイルに保存

        Args:
            config_file (str): 設定ファイルのパス

        Returns:
            bool: 成功した場合はTrue、失敗した場合はFalse
        """
        try:
            # Config オブジェクトを辞書に変換
            config_dict = asdict(self)
            with open(config_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            return True
        except Exception as e:
            print(f"設定ファイルの保存に失敗しました: {str(e)}")
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