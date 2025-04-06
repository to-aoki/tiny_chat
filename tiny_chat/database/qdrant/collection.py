from typing import List, Dict, Any, Optional, TYPE_CHECKING

from qdrant_client.http.models.models import QueryResponse
from tiny_chat.database.embeddings.text_chunk import TextChunker
from tiny_chat.database.qdrant.rag_strategy import RagStrategyFactory

# 循環インポートを避けるための条件付きインポート
if TYPE_CHECKING:
    from tiny_chat.database.qdrant.qdrant_manager import QdrantManager


class Collection:
    """
    Qdrantコレクションを表すクラス。
    コレクションの設定や説明などを保持し、保存・読み込みの操作を提供します。
    """

    def __init__(
        self,
        collection_name: str = "documents",
        description: str = "Default documents",
        qdrant_manager: Optional["QdrantManager"] = None,
        strategy: str = "default",
        chunk_size: Optional[int] = 1024,
        chunk_overlap: Optional[int] = 24,
        top_k: int = 3,
        score_threshold: float = 0.4,
        rag_strategy: Optional[str] = "bm25_static",
        use_gpu: Optional[bool] = False,
    ):
        """
        コレクションの初期化

        Args:
            collection_name: コレクション名
            description: コレクションの説明
            qdrant_manager: QdrantManagerインスタンス（Noneの場合は初期化時に設定しない）
            strategy: 検索戦略の種類
            chunk_size: テキストチャンクの最大サイズ
            chunk_overlap: テキストチャンクのオーバーラップ
            top_k: 検索結果の上位件数
            score_threshold: 検索結果のスコアしきい値
            rag_strategy: 使用するRAG戦略
            use_gpu: GPUを使用するかどうか
        """
        self.collection_name = collection_name
        self.description = description
        self.qdrant_manager = qdrant_manager
        self.strategy = strategy
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rag_strategy = rag_strategy
        self.use_gpu = use_gpu
        self.strategy_obj = RagStrategyFactory.get_strategy(
            strategy_name=rag_strategy, use_gpu=use_gpu)
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # QdrantManagerが与えられている場合はコレクションの存在を確認
        if qdrant_manager:
            self._ensure_collection_exists(qdrant_manager)
    
    def _ensure_collection_exists(self, qdrant_manager=None):
        """
        コレクションが存在することを確認し、存在しない場合は作成します。

        Args:
            qdrant_manager: QdrantManagerインスタンス（Noneの場合はself.qdrant_managerを使用）
        """
        if qdrant_manager is None:
            qdrant_manager = self.qdrant_manager
            
        if qdrant_manager is None:
            return
        # コレクションの存在確認と作成はQdrantManagerに委譲
        qdrant_manager._ensure_collection_exists(self.collection_name)

    def save(self, qdrant_manager=None):
        """
        コレクション情報を"collection_descriptions"コレクションに保存します。

        Args:
            qdrant_manager: QdrantManagerインスタンス（Noneの場合はself.qdrant_managerを使用）

        Returns:
            str: 保存されたレコードのID
        """
        if qdrant_manager is None:
            qdrant_manager = self.qdrant_manager
            
        if qdrant_manager is None:
            raise ValueError("QdrantManagerが設定されていません")
        
        # CollectionDescriptionManagerを使ってコレクション説明を保存
        collection_desc_manager = CollectionDescriptionManager(qdrant_manager)
        
        # コレクション情報をメタデータとして保存
        return collection_desc_manager.add_collection_description(
            collection_name=self.collection_name, 
            description=self.description,
            metadata={
                "strategy": self.strategy,
                "top_k": self.top_k,
                "score_threshold": self.score_threshold,
                "rag_strategy": self.rag_strategy,
                "use_gpu": self.use_gpu,
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap
            }
        )

    @classmethod
    def load(cls, collection_name: str, qdrant_manager: "QdrantManager"):
        """
        "collection_descriptions"コレクションからコレクション情報を読み込みます。

        Args:
            collection_name: 読み込むコレクション名
            qdrant_manager: QdrantManagerインスタンス

        Returns:
            Optional[Collection]: 読み込まれたCollectionインスタンス（見つからない場合はNone）
        """
        # CollectionDescriptionManagerを使ってコレクション説明を読み込み
        collection_desc_manager = CollectionDescriptionManager(qdrant_manager)
        
        # コレクション説明を取得
        result = collection_desc_manager.get_collection_description(collection_name)
        if not result:
            return None
        
        # 取得したメタデータからCollectionインスタンスを生成
        metadata = result.get("metadata", {})
        return cls(
            collection_name=collection_name,
            description=result.get("description", ""),
            qdrant_manager=qdrant_manager,
            strategy=metadata.get("strategy", "default"),
            chunk_size=metadata.get("chunk_size", 1024),
            chunk_overlap=metadata.get("chunk_overlap", 24),
            top_k=metadata.get("top_k", 3),
            score_threshold=metadata.get("score_threshold", 0.4),
            rag_strategy=metadata.get("rag_strategy", "bm25_static"),
            use_gpu=metadata.get("use_gpu", False)
        )
    
    @classmethod
    def list_collections(cls, qdrant_manager: "QdrantManager", limit: int = 100):
        """
        利用可能なすべてのコレクション説明のリストを取得します。

        Args:
            qdrant_manager: QdrantManagerインスタンス
            limit: 取得する最大数

        Returns:
            List[Dict[str, Any]]: コレクション説明のリスト
        """
        collection_desc_manager = CollectionDescriptionManager(qdrant_manager)
        
        return collection_desc_manager.list_collection_descriptions(limit=limit)


class CollectionDescriptionManager:
    """
    Qdrantに格納されているcollectionに対して説明文書をmeta_data、ベクトルとして保存する、
    qdrantのcollection、"collection_descriptions"を操作するクラス
    """

    def __init__(self, qdrant_manager: "QdrantManager"):
        """
        CollectionDescriptionManagerの初期化

        Args:
            qdrant_manager: 使用するQdrantManagerインスタンス
        """
        # 既存のQdrantManagerインスタンスを保存
        self.qdrant_manager = qdrant_manager
        
        # collection_descriptionsコレクションが存在することを確認
        self._ensure_descriptions_collection_exists()
    
    def _ensure_descriptions_collection_exists(self):
        """
        collection_descriptionsコレクションが存在することを確認します。
        """
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # コレクションの存在を確認（内部で必要に応じて作成される）
            self.qdrant_manager._ensure_collection_exists()
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)

    def add_collection_description(
        self, 
        collection_name: str, 
        description: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        コレクションの説明を追加する

        Args:
            collection_name: 説明対象のコレクション名
            description: コレクションの説明文
            metadata: 追加のメタデータ（オプション）

        Returns:
            str: 追加された説明のID
        """
        if metadata is None:
            metadata = {}
        
        # メタデータにコレクション名を追加
        metadata["collection_name"] = collection_name
        
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # 説明文をQdrantに追加
            return self.qdrant_manager.add_document(document=description, metadata=metadata)
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)

    def add_collection_descriptions(
        self, 
        collection_names: List[str], 
        descriptions: List[str], 
        metadata_list: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        """
        複数のコレクション説明を一括で追加する

        Args:
            collection_names: 説明対象のコレクション名のリスト
            descriptions: コレクションの説明文のリスト
            metadata_list: 追加のメタデータのリスト（オプション）

        Returns:
            List[str]: 追加された説明のIDリスト
        """
        if metadata_list is None:
            metadata_list = [{} for _ in collection_names]
        elif len(metadata_list) != len(collection_names):
            raise ValueError("metadata_listの長さはcollection_namesと一致する必要があります")
        
        # 各メタデータにコレクション名を追加
        for i, collection_name in enumerate(collection_names):
            metadata_list[i]["collection_name"] = collection_name
        
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # 説明文をQdrantに一括追加
            return self.qdrant_manager.add_documents(documents=descriptions, metadata_list=metadata_list)
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)

    def search_collections(self, query: str, top_k: int = 5, score_threshold: float = 0.4) -> List[QueryResponse]:
        """
        クエリに基づいてコレクションを検索する

        Args:
            query: 検索クエリ
            top_k: 返す結果の数
            score_threshold: スコアのしきい値

        Returns:
            List[QueryResponse]: 検索結果
        """
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # 検索を実行
            return self.qdrant_manager.query_points(
                query=query,
                top_k=top_k,
                score_threshold=score_threshold
            )
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)

    def get_collection_description(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """
        特定のコレクションの説明を取得する

        Args:
            collection_name: コレクション名

        Returns:
            Optional[Dict[str, Any]]: コレクションの説明情報（見つからない場合はNone）
        """
        # コレクション名に一致するフィルタを作成
        filter_params = {"collection_name": collection_name}
        
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # 完全一致検索を実行
            results = self.qdrant_manager.query_points(
                query="",  # 空のクエリでフィルタのみで検索
                top_k=1,
                score_threshold=0.0,
                filter_params=filter_params
            )
            
            if results and len(results) > 0:
                # 最初の結果を返す
                result = results[0]
                return {
                    "id": result.id,
                    "description": result.payload.get("text", ""),
                    "metadata": {k: v for k, v in result.payload.items() if k != "text"}
                }
            
            return None
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)

    def update_collection_description(
        self, 
        collection_name: str, 
        new_description: str, 
        new_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        コレクションの説明を更新する

        Args:
            collection_name: 更新対象のコレクション名
            new_description: 新しい説明文
            new_metadata: 新しいメタデータ（オプション）

        Returns:
            bool: 更新に成功したかどうか
        """
        # 既存の説明を取得
        existing = self.get_collection_description(collection_name)
        if not existing:
            # 存在しない場合は新規追加
            self.add_collection_description(collection_name, new_description, new_metadata)
            return True
        
        # 既存のIDを取得
        existing_id = existing.get("id")
        
        # 既存のメタデータをベースに新しいメタデータを作成
        metadata = existing.get("metadata", {}).copy()
        if new_metadata:
            metadata.update(new_metadata)
        
        # コレクション名は保持
        metadata["collection_name"] = collection_name
        
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # 一旦削除してから追加し直す（更新に相当）
            self.qdrant_manager.delete_by_filter({"id": existing_id})
            
            # 新しい説明を追加
            self.qdrant_manager.add_document(document=new_description, metadata=metadata)
            
            return True
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)

    def delete_collection_description(self, collection_name: str) -> bool:
        """
        コレクションの説明を削除する

        Args:
            collection_name: 削除対象のコレクション名

        Returns:
            bool: 削除に成功したかどうか
        """
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # フィルタを使って削除
            operation_id = self.qdrant_manager.delete_by_filter({"collection_name": collection_name})
            
            # operation_idが返ってきたら成功とみなす
            return operation_id is not None
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)

    def list_collection_descriptions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        すべてのコレクション説明のリストを取得する

        Args:
            limit: 取得する最大数

        Returns:
            List[Dict[str, Any]]: コレクション説明のリスト
        """
        # 元のコレクション名を一時保存
        original_collection_name = self.qdrant_manager.collection_name
        
        try:
            # collection_descriptionsコレクションを使用するように一時的に変更
            self.qdrant_manager.set_collection_name("collection_descriptions")
            
            # scrollメソッドを使用してデータを取得
            results = []
            offset = None
            batch_size = min(1000, limit)
            
            while True:
                batch, next_offset = self.qdrant_manager.client.scroll(
                    collection_name="collection_descriptions",
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )
                
                if not batch:
                    break
                
                for point in batch:
                    collection_name = point.payload.get("collection_name")
                    if collection_name:
                        results.append({
                            "id": point.id,
                            "collection_name": collection_name,
                            "description": point.payload.get("text", ""),
                            "metadata": {k: v for k, v in point.payload.items() 
                                        if k not in ["text", "collection_name"]}
                        })
                
                if len(batch) < batch_size or len(results) >= limit or next_offset is None:
                    break
                
                offset = next_offset
            
            return results
        except Exception as e:
            return []
        finally:
            # 元のコレクション名に戻す
            self.qdrant_manager.set_collection_name(original_collection_name)


# テスト用のメイン処理は、実行時にのみQdrantManagerをインポートするように修正
if __name__ == "__main__":
    # 使用例 - 実行時のみインポート
    from tiny_chat.database.qdrant.qdrant_manager import QdrantManager
    
    # QdrantManagerを初期化
    qdrant_manager = QdrantManager(file_path="./qdrant_data")
    
    # 方法1: Collectionクラスを使用して新しいコレクション説明を作成・保存
    documents_collection = Collection(
        collection_name="documents",
        description="ドキュメントを保存するコレクションです。文書の本文とメタデータを格納します。",
        qdrant_manager=qdrant_manager,
        top_k=5,
        score_threshold=0.5
    )
    documents_collection.save()
    
    users_collection = Collection(
        collection_name="users", 
        description="ユーザー情報を保存するコレクションです。ユーザープロファイルを格納します。",
        qdrant_manager=qdrant_manager
    )
    users_collection.save()
    
    # 方法2: Collection.loadを使用して既存のコレクション説明を読み込む
    loaded_collection = Collection.load("documents", qdrant_manager)
    if loaded_collection:
        print(f"読み込まれたコレクション: {loaded_collection.collection_name}")
        print(f"説明: {loaded_collection.description}")
        print(f"top_k: {loaded_collection.top_k}")
    
    # 方法3: コレクション一覧を取得
    collections = Collection.list_collections(qdrant_manager)
    print("\n利用可能なコレクション:")
    for coll in collections:
        print(f"- {coll['collection_name']}: {coll['description']}")
    
    # 方法4: CollectionDescriptionManagerを直接使用する例
    desc_manager = CollectionDescriptionManager(qdrant_manager)
    
    # コレクション説明の検索
    results = desc_manager.search_collections("ドキュメント")
    print("\n検索結果:")
    for result in results:
        print(f"- {result.payload.get('collection_name')}: {result.payload.get('text')[:50]}...")
