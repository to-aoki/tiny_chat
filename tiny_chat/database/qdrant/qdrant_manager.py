import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models.models import QueryResponse

try:
    from tiny_chat.database.embeddings.text_chunk import TextChunker
    from tiny_chat.database.qdrant.rag_strategy import RagStrategyFactory, RAGStrategy
    from tiny_chat.database.qdrant.collection import Collection
except:
    import os
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
    from tiny_chat.database.embeddings.text_chunk import TextChunker
    from tiny_chat.database.qdrant.rag_strategy import RagStrategyFactory, RAGStrategy
    from tiny_chat.database.qdrant.collection import Collection


class QdrantManager:
    """
    Qdrantベクターデータベースとの連携を管理するクラス
    """

    def __init__(self,
        collection_name: str = "default",
        file_path: str = "./qdrant_data",
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        chunk_size: Optional[int] = 2000,
        chunk_overlap: Optional[int] = 0,
        top_k: int = 3,
        score_threshold: float = 0.2,
        rag_strategy: Optional[str] = "bm25",
        use_gpu: Optional[bool] = False,
        **kwargs
    ):

        selected_collection_name = kwargs.get("selected_collection_name", None)
        if selected_collection_name is not None:
            collection_name = selected_collection_name

        self.server_url = None
        self.api_key = None
        self.file_path = None
        if server_url:
            # サーバーに接続
            if server_url.startswith("http"):
                self.client = QdrantClient(url=server_url, api_key=api_key)
            elif server_url.startswith("dns://"):
                from urllib.parse import urlparse
                # dns://localhost:6334 のような dns://から始まるURIはgRPCとみなす
                parsed_uri = urlparse(server_url)
                self.client = QdrantClient(host=parsed_uri.hostname, port=parsed_uri.port, prefer_grpc=True)
            self.server_url = server_url
            self.api_key = api_key
        elif file_path == ":memory:":
            # メモリモード - ファイルを使わない
            self.client = QdrantClient(":memory:")
            self.file_path = file_path
        else:
            # ローカルファイルモード
            self.client = QdrantClient(path=file_path)
            self.file_path = file_path

        self.collection_name = collection_name
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.rag_strategy = rag_strategy
        self.use_gpu = use_gpu
        self.strategy = RagStrategyFactory.get_strategy(
            strategy_name=rag_strategy, use_gpu=use_gpu)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        Collection.ensure_collection_descriptions_exists(qdrant_manager=self)
        collection = Collection.load(self.collection_name, qdrant_manager=self)
        if collection is None:
            collection = Collection(
                collection_name=self.collection_name,
                top_k=top_k,
                score_threshold=score_threshold,
                rag_strategy=rag_strategy,
                use_gpu=use_gpu,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            collection.save(qdrant_manager=self)

    def is_need_reconnect(
        self,
        file_path: str = None,
        server_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **kwargs
    ):
        if server_url != self.server_url:
            return True
        if self.server_url is not None and api_key != self.api_key:
            return True
        if file_path != self.file_path:
            return True
        return False

    def update_settings(
        self,
        collection_name: str = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        rag_strategy: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        **kwargs
    ):
        selected_collection_name = kwargs.get("selected_collection_name", None)
        if selected_collection_name is not None:
            collection_name = selected_collection_name
        if collection_name is not None and self.collection_name != collection_name:
            self.set_collection_name(collection_name=collection_name)
        if chunk_size is not None and self.chunk_size != chunk_size:
            self.chunk_size = chunk_size
        if chunk_overlap is not None and self.chunk_overlap != chunk_overlap:
            self.chunk_overlap = chunk_overlap
        self.top_k = top_k if top_k is not None else self.top_k
        self.score_threshold = score_threshold if score_threshold is not None else self.score_threshold
        if (use_gpu is not None and self.use_gpu != use_gpu) or (
                rag_strategy is not None and self.rag_strategy != rag_strategy):
            update_rag_strategy = self.rag_strategy if rag_strategy else self.rag_strategy
            update_use_gpu = self.use_gpu if use_gpu else self.use_gpu
            update_strategy = RagStrategyFactory.get_strategy(
                strategy_name=update_rag_strategy, use_gpu=update_use_gpu
            )
            if update_strategy is None:
                raise ValueError(f"{update_rag_strategy} not found.")
            self.strategy = update_strategy
            self.rag_strategy = update_rag_strategy
            self.use_gpu = update_use_gpu

    def ensure_collection_exists(
            self, collection_name: Optional[str] = None, strategy: Optional[RAGStrategy] = None):
        """
        コレクションが存在するか確認し、なければ作成する

        Args:
            collection_name: コレクション名（Noneの場合はself.collection_nameを使用）
        """
        # コレクション名を確定（引数がNoneの場合はインスタンス変数を使用）
        collection_name = collection_name if collection_name is not None else self.collection_name

        # コレクション一覧を取得
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]

        if strategy is None:
            strategy = self.strategy

        # コレクションが存在しない場合は新規作成
        if collection_name not in collection_names:
            # コレクションの作成
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=strategy.create_vector_config(),
                sparse_vectors_config=strategy.create_sparse_vectors_config(),
                quantization_config=models.ScalarQuantization(
                    scalar=models.ScalarQuantizationConfig(
                        type=models.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            )

    def get_collection(self, collection_name: str) -> Any:
        """
        コレクション情報を取得する

        Args:
            collection_name: コレクション名

        Returns:
            Any: コレクション情報
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections if c != Collection.STORED_COLLECTION_NAME]
        if collection_name not in collection_names:
            return None

        return self.client.get_collection(collection_name=collection_name)

    def add(self, collection_name: str, **kwargs):
        """
        文書をコレクションに追加する

        Args:
            collection_name: コレクション名
            **kwargs: 追加する文書とメタデータ
        """
        # self.ensure_collection_exists(collection_name)
        
        documents = kwargs.get("texts", [])
        metadatas = kwargs.get("metadatas", [])
        ids = kwargs.get("ids", [])

        # 文書数を確認
        if not documents:
            raise ValueError("追加する文書が指定されていません")

        # metadataがない場合は空のメタデータを使用
        if not metadatas:
            metadatas = [{} for _ in documents]

        # idsがある場合はメタデータにidを追加
        if ids and len(ids) == len(documents):
            for i, doc_id in enumerate(ids):
                metadatas[i]["id"] = doc_id

        # 文書を追加
        return self.add_documents(documents, metadatas, collection_name)

    def add_document(
        self,
        document: str,
        metadata: Dict[str, Any],
        collection_name: Optional[str] = None,
        use_chunker: bool = True,
        strategy: Optional[RAGStrategy] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> str:
        """
        単一の文書をチャンク分割してベクトル化しQdrantに追加する

        Args:
            document: 文書テキスト
            metadata: 文書のメタデータ（参照元、ページ番号、URL等）
            collection_name: コレクション名（Noneの場合はself.collection_nameを使用）

        Returns:
            str: 追加された文書のID
        """
        result = self.add_documents([document], [metadata], collection_name, use_chunker,
                                    strategy=strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return result[0] if result else None
        
    def add_documents(
        self,
        documents: List[str],
        metadata_list: List[Dict[str, Any]],
        collection_name: Optional[str] = None,
        use_chunker: bool = True,
        strategy: Optional[RAGStrategy] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[str]:
        """
        複数の文書をチャンク分割してベクトル化しQdrantに追加する

        Args:
            documents: 文書テキストのリスト
            metadata_list: 各文書のメタデータのリスト（参照元、ページ番号、URL等）
            collection_name: コレクション名（Noneの場合はself.collection_nameを使用）

        Returns:
            List[str]: 追加された文書のIDリスト（元の文書ごとに1つのID）
        """
        # コレクション名を確定
        collection_name = collection_name if collection_name is not None else self.collection_name

        if strategy is None:
            strategy = self.strategy

        if len(documents) != len(metadata_list):
            raise ValueError("documents と metadata_list の長さが一致しません")

        points = []
        ids = []
        original_ids = []

        # 各文書を処理
        for i, (doc, metadata) in enumerate(zip(documents, metadata_list)):
            # メタデータからIDを取得または生成
            original_id = metadata.get("id")
            
            # 元のIDを保存
            if original_id:
                original_ids.append(original_id)
            else:
                # 新規IDを生成
                original_id = str(uuid.uuid4())
                original_ids.append(original_id)
            
            # 文書をチャンク分割
            if use_chunker:
                if chunk_size is None:
                    chunk_size = self.chunk_size
                if chunk_overlap is None:
                    chunk_overlap = self.chunk_overlap
                doc_chunks = TextChunker.split_text(
                    doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                # 各チャンクを処理
                for chunk_idx, chunk in enumerate(doc_chunks):
                    if original_id:
                        chunk_id = f"{original_id}_chunk_{chunk_idx}"
                        # UUIDでない場合は元のIDをハッシュしてUUIDを生成
                        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
                    else:
                        # IDがない場合は新規生成
                        point_id = str(uuid.uuid4())

                    # チャンク用メタデータをコピー
                    chunk_metadata = metadata.copy()
                    chunk_metadata["chunk_index"] = chunk_idx
                    chunk_metadata["chunk_total"] = len(doc_chunks)

                    if "id" in chunk_metadata and chunk_metadata["id"] != point_id:
                        chunk_metadata["parent_id"] = chunk_metadata["id"]

                    chunk_metadata["id"] = point_id
                    # Qdrantポイントの作成
                    point = models.PointStruct(
                        id=point_id,
                        vector=strategy.vector(chunk),
                        payload={
                            "text": chunk,
                            **chunk_metadata
                        }
                    )
                    points.append(point)

                    # 最初のチャンクのIDを記録（元の文書のIDとして使用）
                    if chunk_idx == 0:
                        ids.append(point_id)

            else:
                point_id = str(uuid.uuid4())
                # Qdrantポイントの作成
                point = models.PointStruct(
                    id=point_id,
                    vector=strategy.vector(doc),
                    payload={
                        "text": doc,
                        **metadata
                    }
                )
                points.append(point)

        # ポイントをQdrantに追加
        if points:
            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

        # 元の文書IDを返す
        return original_ids

    def query_points(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.4,
        collection_name: Optional[str] = None,
        filter_params: Optional[Dict[str, Any]] = None,
        strategy : Optional[RAGStrategy] = None,
        dense_text: Optional[str] = None,
    ) -> List[QueryResponse]:
        """
        クエリに基づいて文書を検索する (ハイブリッド検索)

        Args:
            query: 検索クエリ
            top_k: 返す結果の数
            score_threshold: スコアのしきい値
            collection_name: コレクション名（Noneの場合はself.collection_nameを使用）
            filter_params: 検索フィルタ（参照元、ページ番号等でフィルタリング）

        Returns:
            List[QueryResponse]: 検索結果 (QueryResponseオブジェクトのリスト)
        """
        if collection_name is None:
            collection_name = self.collection_name

        if strategy is None:
            strategy = self.strategy

        # 検索フィルタを作成
        search_filter = None
        if filter_params and self.server_url is not None:
            filter_conditions = []
            for key, value in filter_params.items():
                if value:  # 値が空でない場合のみフィルタに追加
                    if isinstance(value, list):
                        if value:  # リストが空でない場合
                            filter_conditions.append(
                                models.FieldCondition(
                                    key=key,
                                    match=models.MatchAny(any=value)
                                )
                            )
                    else:
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchValue(value=value)
                            )
                        )

            if filter_conditions:
                search_filter = models.Filter(
                    must=filter_conditions
                )

        if top_k is None:
            top_k = self.top_k
        if score_threshold is None:
            score_threshold = self.score_threshold

        if hasattr(strategy, 'rerank'):
            # 2倍にする
            top_k = top_k * 2

        prefetch = strategy.prefetch(query, top_k, dense_text)
        if prefetch:
            response = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=strategy.query(query, dense_text),
                limit=top_k,
                with_vectors=False,
                with_payload=True,
                query_filter=search_filter,
                # score_threshold=score_threshold # 効いてなさそう (qdrant-client 1.13.3 file)
            )
        else:
            response = self.client.query_points(
                collection_name=collection_name,
                query=strategy.query(query, dense_text),
                using=strategy.use_vector_name(),
                limit=top_k,
                with_vectors=False,
                with_payload=True,
                query_filter=search_filter,
                # score_threshold=score_threshold
            )

        points = response.points
        results = []
        for point in points:
            if filter_params and self.server_url is None:
                not_match = False
                for key, value in filter_params.items():
                    if isinstance(value, list):
                        if value not in point.payload[key]:
                            not_match = True
                            break
                    else:
                        if value != point.payload[key]:
                            not_match = True
                            break
                if not_match:
                    continue
            if score_threshold < point.score:
                results.append(point)

        if hasattr(strategy, 'rerank'):
            if results is not None or len(results) > 2:
                return strategy.rerank(query, results, top_k, score_threshold)

        return results

    def set_collection_name(self, collection_name: str, force_create=False) -> None:
        """
        現在のデフォルトコレクション名を変更する
        
        Args:
            collection_name: 新しいコレクション名
        """
        self.collection_name = collection_name
        if force_create:
            self.ensure_collection_exists(collection_name)
    
    def get_collections(self) -> List[str]:
        """
        利用可能なコレクション一覧を取得する

        Returns:
            List[str]: コレクション名のリスト
        """
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        return sorted(collection_names)
        
    def get_sources(self, collection_name: Optional[str] = None,
                    limit=1000,
                    source_name:Optional[str] = "source"
    ) -> List[str]:
        """
        データベース内のすべての参照元（ソース）を取得する

        Args:
            collection_name: コレクション名（Noneの場合はself.collection_nameを使用）
            limit: 取得する最大ソース数

        Returns:
            List[str]: ユニークな参照元のリスト
        """
        # コレクション名を確定
        collection_name = collection_name if collection_name is not None else self.collection_name
        
        # コレクションの存在を確認
        # self.ensure_collection_exists(collection_name)

        try:
            # scrollメソッドはbatch数を返すのでイテレーションが必要
            sources = set()
            offset = None  # Noneを使うとスクロール開始地点から始まる
            batch_size = min(1000, limit)

            # 最初は全ペイロードを取得して処理
            while True:
                batch, next_offset = self.client.scroll(
                    collection_name=collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,  # 全ペイロードを取得
                    with_vectors=False
                )

                if not batch:
                    break

                for point in batch:
                    if point.payload and source_name in point.payload:
                        source = point.payload.get(source_name)
                        if source:
                            sources.add(source)

                if len(batch) < batch_size or len(sources) >= limit or next_offset is None:
                    break

                offset = next_offset
                
            return sorted(list(sources))
        except Exception as e:
            return []

    def count_documents(self, collection_name: Optional[str] = None) -> int:
        """
        コレクション内の文書数を取得する

        Args:
            collection_name: コレクション名（Noneの場合はself.collection_nameを使用）

        Returns:
            int: 文書の総数
        """
        # コレクション名を確定
        collection_name = collection_name if collection_name is not None else self.collection_name
        
        # コレクションの存在を確認
        # self.ensure_collection_exists(collection_name)

        collection_info = self.client.get_collection(collection_name=collection_name)
        return collection_info.points_count
        
    def delete_collection(self, collection_name: Optional[str] = None) -> bool:
        """
        コレクションを削除する

        Args:
            collection_name: 削除するコレクション名 (Noneの場合は現在のコレクションを削除)

        Returns:
            bool: 削除に成功したかどうか
        """
        # コレクション名を確定
        collection_name = collection_name if collection_name is not None else self.collection_name
            
        try:
            # コレクションが存在するか確認
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections if c.name != Collection.STORED_COLLECTION_NAME]

            if len(collection_names) <= 1 or collection_name not in collection_names:
                return False  # 削除対象のコレクションが存在しない

            # コレクションを削除
            self.client.delete_collection(collection_name=collection_name)

            collection_names.remove(collection_name)
            
            if self.collection_name == collection_name:
                self.collection_name = collection_names[0]

            return True
        except Exception as e:
            return False

    def delete_by_filter(self, filter_params: Dict[str, Any], collection_name: Optional[str] = None) -> int:
        """
        フィルタに一致する文書を削除する

        Args:
            filter_params: 削除フィルタ
            collection_name: コレクション名（Noneの場合はself.collection_nameを使用）

        Returns:
            int: 削除された文書の数
        """
        # コレクション名を確定
        collection_name = collection_name if collection_name is not None else self.collection_name

        filter_conditions = []
        for key, value in filter_params.items():
            if value:  # 値が空でない場合のみフィルタに追加
                if isinstance(value, list):
                    if value:  # リストが空でない場合
                        filter_conditions.append(
                            models.FieldCondition(
                                key=key,
                                match=models.MatchAny(any=value)
                            )
                        )
                else:
                    filter_conditions.append(
                        models.FieldCondition(
                            key=key,
                            match=models.MatchValue(value=value)
                        )
                    )

        if not filter_conditions:
            return 0

        delete_filter = models.Filter(
            must=filter_conditions
        )
        result = self.client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(
                filter=delete_filter
            )
        )
        return result.operation_id


if __name__ == "__main__":
    import argparse
    import time

    # コマンドライン引数の設定
    parser = argparse.ArgumentParser()
    # 必須引数
    parser.add_argument(
        "--strategy", "-s",
        type=str,
        default="bm25_static"
    )

    parser.add_argument(
        "--use_gpu", "-g",
        action='store_true',
    )
    args = parser.parse_args()

    start_time = time.time()
    test_path = "./qdrant_test_data"
    manager = QdrantManager(
        collection_name="test",
        rag_strategy=args.strategy, use_gpu=args.use_gpu, file_path=test_path)

    print(f"初期化時間: {time.time() - start_time:.4f}秒")

    # テスト用の文書とメタデータを作成
    documents = [
        "東京は日本の首都で、人口が最も多い都市です。多くの観光名所や企業の本社があります。",
        "大阪は日本の西部に位置する商業都市で、美味しい食べ物と陽気な人々で知られています。",
        "京都は歴史的建造物が多い古都で、多くの神社仏閣や伝統文化が残っています。",
        "北海道は日本最北の島で、雄大な自然と美味しい海産物で有名です。冬はスキーやスノーボードが楽しめます。",
        "沖縄は日本最南端の県で、美しいビーチとサンゴ礁に囲まれた島々からなります。独自の文化も魅力です。"
    ]
    
    metadata_list = [
        {"source": "観光ガイド", "page": 1, "category": "都市", "region": "関東"},
        {"source": "観光ガイド", "page": 2, "category": "都市", "region": "関西"},
        {"source": "観光ガイド", "page": 3, "category": "都市", "region": "関西"},
        {"source": "観光ガイド", "page": 4, "category": "自然", "region": "北海道"},
        {"source": "観光ガイド", "page": 5, "category": "自然", "region": "沖縄"}
    ]
    
    print("=== 文書の追加 ===")
    start_time = time.time()
    ids = manager.add_documents(documents, metadata_list)
    print(f"ドキュメント追加時間: {time.time() - start_time:.4f}秒")
    print(f"追加された文書ID: {ids}")
    print(f"コレクション内の文書数: {manager.count_documents()}")
    
    # 検索テスト
    print("\n=== 検索テスト ===")
    queries = [
        "日本の首都はどこですか？",
        "美味しい食べ物がある都市は？",
        "歴史的な観光地はどこですか？",
        "自然を楽しめる場所はどこですか？"
    ]
    
    for query in queries:
        print(f"\nクエリ: {query}")
        start_time = time.time()
        results = manager.query_points(query, top_k=3, score_threshold=0.)
        
        print(f"検索結果（上位{len(results)}件）:")
        for i, result in enumerate(results):
            print(f"  {i+1}. スコア: {result.score:.4f}")
            print(f"     文書: {result.payload.get('text')}")
            print(f"     メタデータ: {', '.join([f'{k}={v}' for k, v in result.payload.items() if k != 'text'])}")
    
    # フィルタリングテスト
    print("\n=== フィルタリングテスト ===")
    query = "観光が楽しめるところは？"
    filter_params = {"region": "関西"}
    
    print(f"クエリ: {query}, フィルタ: {filter_params}")
    results = manager.query_points(query, top_k=3, filter_params=filter_params, score_threshold=-1)

    print(manager.get_sources())

    print(f"検索結果（上位{len(results)}件）:")
    for i, result in enumerate(results):
        print(f"  {i+1}. スコア: {result.score:.4f}")
        print(f"     文書: {result.payload.get('text')}")
        print(f"     メタデータ: {', '.join([f'{k}={v}' for k, v in result.payload.items() if k != 'text'])}")

    manager.client.close()

    import shutil
    shutil.rmtree(test_path)