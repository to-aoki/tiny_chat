from typing import Optional
from tiny_chat.database.qdrant.rag_strategy import NoopRAGStrategy


class Collection:

    STORED_COLLECTION_NAME = "collection_descriptions"

    """
    Qdrantコレクションを表すクラス。
    コレクションの設定や説明などを保持し、保存・読み込みの操作を提供します。
    """

    def __init__(
        self,
        collection_name: str = "defaults",
        description: str = "Default documents",
        chunk_size: Optional[int] = 1024,
        chunk_overlap: Optional[int] = 24,
        top_k: int = 3,
        score_threshold: float = 0.4,
        rag_strategy: Optional[str] = "bm25_static",
        use_gpu: Optional[bool] = False,
        **kwargs
    ):
        """
        コレクションの初期化

        Args:
            collection_name: コレクション名
            description: コレクションの説明
            qdrant_manager: QdrantManagerインスタンス（Noneの場合は初期化時に設定しない）
            chunk_size: テキストチャンクの最大サイズ
            chunk_overlap: テキストチャンクのオーバーラップ
            top_k: 検索結果の上位件数
            score_threshold: 検索結果のスコアしきい値
            rag_strategy: 使用するRAG戦略
            use_gpu: GPUを使用するかどうか
        """
        self.collection_name = collection_name
        self.description = description
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rag_strategy = rag_strategy
        self.use_gpu = use_gpu
        self.rag_strategy = rag_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def save(self, qdrant_manager=None):
        """
        コレクション情報を"collection_descriptions"コレクションに保存します。

        Args:
            qdrant_manager: QdrantManagerインスタンス（Noneの場合はself.qdrant_managerを使用）

        Returns:
            str: 保存されたレコードのID
        """
        if qdrant_manager is None:
            raise ValueError("QdrantManagerが設定されていません")

        current_strategy = qdrant_manager.rag_strategy
        qdrant_manager.rag_strategy = NoopRAGStrategy()
        qdrant_manager.ensure_collection_exists(self.STORED_COLLECTION_NAME)

        # コレクション情報をメタデータとして保存
        qdrant_manager.add_document(
            collection_name=self.STORED_COLLECTION_NAME,
            document=self.description,
            metadata={
                "collection_name": self.collection_name,
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "rag_strategy": self.rag_strategy,
                "use_gpu": self.use_gpu,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap
            },
            use_chunker=False
        )
        qdrant_manager.rag_strategy = current_strategy
        qdrant_manager.ensure_collection_exists(self.collection_name)


    @classmethod
    def load(cls, collection_name: str, qdrant_manager=None):
        """
        "collection_descriptions"コレクションからコレクション情報を読み込みます。

        Args:
            collection_name: 読み込むコレクション名
            qdrant_manager: QdrantManagerインスタンス

        Returns:
            Optional[Collection]: 読み込まれたCollectionインスタンス（見つからない場合はNone）
        """

        # フィルタを使用して特定のコレクション情報を検索
        filter_params = {"collection_name": collection_name}
        results = qdrant_manager.query_points(
            query="",  # 空のクエリで全件取得
            filter_params=filter_params,
            collection_name=cls.STORED_COLLECTION_NAME,
            top_k=1  # 最新の1件だけ取得
        )

        if not results:
            return None

        # 最初の結果を使用
        result = results[0]
        payload = result.payload
        
        # ペイロードからCollectionインスタンスを生成
        return cls(
            collection_name=payload.get("collection_name", collection_name),
            description=payload.get("text", ""),
            chunk_size=payload.get("chunk_size", 1024),
            chunk_overlap=payload.get("chunk_overlap", 24),
            top_k=payload.get("top_k", 3),
            score_threshold=payload.get("score_threshold", 0.4),
            rag_strategy=payload.get("rag_strategy", "bm25_static"),
            use_gpu=payload.get("use_gpu", False)
        )

    @classmethod
    def ensure_collection_descriptions_exists(cls, qdrant_manager):
        current_strategy = qdrant_manager.rag_strategy
        qdrant_manager.rag_strategy = NoopRAGStrategy()
        qdrant_manager.ensure_collection_exists(cls.STORED_COLLECTION_NAME)
        qdrant_manager.rag_strategy = current_strategy
