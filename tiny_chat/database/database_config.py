import json
import os

# デフォルトの設定ファイルパス
DEFAULT_CONFIG_PATH = "database_config.json"


class DatabaseConfig:
    """
    データベースアプリケーションの設定を管理するクラス
    """

    def __init__(
        self,
        file_path: str = "./qdrant_data",
        server_url: str = None,
        api_key: str = None,
        chunk_size: int = 1024,
        chunk_overlap: int = 24,
        top_k: int = 3,
        score_threshold: float = 0.4,
        selected_collection_name: str = "default",
        rag_strategy: str = 'bm25_ruri_xsmall',
        ues_gpu: bool = False,
        **kwargs
    ):
        self.file_path = file_path
        self.server_url = server_url
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.selected_collection_name = selected_collection_name
        self.rag_strategy = rag_strategy
        self.use_gpu = ues_gpu

    @classmethod
    def load(cls, file_path: str) -> 'DatabaseConfig':
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
        config_data = {
            'file_path': self.file_path,
            'server_url': self.server_url,
            'api_key': self.api_key,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'top_k': self.top_k,
            'score_threshold': self.score_threshold,
            'selected_collection_name': self.selected_collection_name,
            'rag_strategy': self.rag_strategy,
            'use_gpu': self.use_gpu,
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)

