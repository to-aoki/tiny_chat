from typing import List, Dict, Tuple, Union, Iterable, Optional, Any, Sequence

from sudachipy import tokenizer
from sudachipy import dictionary
import stopwordsiso
from fastembed import SparseEmbedding, SparseTextEmbedding
from fastembed.common import OnnxProvider


class BM25TextEmbedding(SparseTextEmbedding):

    def __init__(
        self,
        model_name: str = "Qdrant/bm25",
        cache_dir: Optional[str] = None,
        threads: Optional[int] = None,
        providers: Optional[Sequence[OnnxProvider]] = None,
        cuda: bool = False,
        device_ids: Optional[list[int]] = None,
        lazy_load: bool = False,
        language: str = "japanese",
        k: float = 1.2,
        b: float = 0.75,
        avg_len: float = 256.0,
        token_max_length: int = 40,
        **kwargs: Any,
    ):
        """BM25TextEmbeddingを初期化します。

        Args:
            model_name (str): 使用するモデル名。デフォルトは"Qdrant/bm25"
            cache_dir (Optional[str]): キャッシュディレクトリ
            threads (Optional[int]): 使用するスレッド数
            providers (Optional[Sequence[OnnxProvider]]): ONNXプロバイダー
            cuda (bool): CUDAを使用するかどうか
            device_ids (Optional[list[int]]): デバイスID
            lazy_load (bool): 遅延ロードを使用するかどうか
            **kwargs: 追加のパラメータ
        """
        self.is_japanese = True
        if language != "japanese":
            kwargs["language"] = language
            self.is_japanese = False

        kwargs["k"] = k
        kwargs["b"] = b
        kwargs["avg_len"] = avg_len
        kwargs["token_max_length"] = token_max_length

        # SparseTextEmbeddingのコンストラクタを呼び出し
        super().__init__(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=threads,
            providers=providers,
            cuda=cuda,
            device_ids=device_ids,
            lazy_load=lazy_load,
            disable_stemmer=True if self.is_japanese else False,  # 日本語は機能しない
            **kwargs
        )

        # 日本語処理のための追加コンポーネント
        if self.is_japanese:
            self._tokenizer = dictionary.Dictionary().create()
            self._tokenizer_mode = tokenizer.Tokenizer.SplitMode.C  # 最も分割単位が細かいモードを使用
            self._stopwords = stopwordsiso.stopwords("ja")

    def _remove_symbols(self, morphemes: List) -> List:
        """補助記号を削除します。

        Args:
            morphemes (List): トークン化された形態素のリスト

        Returns:
            List: 記号を含まない形態素のリスト
        """
        return [morph for morph in morphemes if morph.part_of_speech()[0] != "補助記号"]

    def _remove_stopwords(self, morphemes: List) -> List:
        """ストップワードを削除します。

        Args:
            morphemes (List): トークン化された形態素のリスト

        Returns:
            List: ストップワードを含まない形態素のリスト
        """
        return [morph for morph in morphemes if morph.surface() not in self._stopwords]

    def _tokenize(self, text: str) -> List[str]:
        """Sudachiを使用してテキストをトークン化します。

        Args:
            text (str): トークン化する入力テキスト

        Returns:
            List[str]: トークンのリスト
        """
        try:
            # Sudachiでの形態素解析
            morphemes = self._tokenizer.tokenize(text, self._tokenizer_mode)
            
            # 補助記号を削除
            morphemes = self._remove_symbols(morphemes)
            
            # ストップワードを削除
            morphemes = self._remove_stopwords(morphemes)
            
            # 表層形を取得
            return [morph.surface() for morph in morphemes]
        except Exception as e:
            return []

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """ドキュメントに対する埋め込みを生成します。
        日本語向けの前処理を行ってから親クラスのembedを呼び出します。

        Args:
            documents: 埋め込みを生成するテキストまたはテキストのリスト
            batch_size: エンコーディングのバッチサイズ
            parallel: 並列処理の使用（0: 全コア使用、None: 使用しない）
            **kwargs: 追加のパラメータ

        Returns:
            Iterable[SparseEmbedding]: 埋め込みのシーケンス
        """
        if self.is_japanese:
            # 単一文書の場合リストに変換
            if isinstance(documents, str):
                documents = [documents]

            filtered_documents = []
            for doc in documents:
                tokens = self._tokenize(text=doc)
                concat_tokens = " ".join(tokens)
                filtered_documents.append(concat_tokens)
        else:
            filtered_documents = documents
        
        # 親クラスのembedメソッドを呼び出す
        return super().embed(documents=filtered_documents, batch_size=batch_size, parallel=parallel, **kwargs)

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        """クエリに対する埋め込みを生成します。
        日本語向けの前処理を行ってから親クラスのquery_embedを呼び出します。

        Args:
            query: 埋め込みを生成するクエリテキストまたはクエリのリスト
            **kwargs: 追加のパラメータ

        Returns:
            Iterable[SparseEmbedding]: クエリの埋め込み
        """
        # 単一クエリの場合リストに変換
        if self.is_japanese:
            if isinstance(query, str):
                tokens = self._tokenize(text=query)
                tokenized_query = " ".join(tokens)
                return super().query_embed(query=tokenized_query, **kwargs)
            else:
                # イテラブルの場合、各クエリに対して前処理を適用
                tokenized_queries = []
                for q in query:
                    tokens = self._tokenize(text=q)
                    tokenized_queries.append(" ".join(tokens))
                return super().query_embed(query=tokenized_queries, **kwargs)
        else:
            return super().query_embed(query=query, **kwargs)

    def calculate_similarity(self, query_embedding: SparseEmbedding, document_embedding: SparseEmbedding) -> float:
        """クエリ埋め込みとドキュメント埋め込み間の類似度を計算します。

        Args:
            query_embedding (SparseEmbedding): クエリの埋め込み
            document_embedding (SparseEmbedding): ドキュメントの埋め込み

        Returns:
            float: 類似度スコア
        """
        # スパース埋め込みの類似度計算方法
        # インデックスとバリューを取得
        q_indices = query_embedding.indices
        q_values = query_embedding.values
        d_indices = document_embedding.indices
        d_values = document_embedding.values
        
        # ドット積を計算
        score = 0.0
        q_idx_dict = {idx: val for idx, val in zip(q_indices, q_values)}
        d_idx_dict = {idx: val for idx, val in zip(d_indices, d_values)}
        
        # 共通するインデックスでスコアを計算
        common_indices = set(q_indices).intersection(set(d_indices))
        for idx in common_indices:
            score += q_idx_dict[idx] * d_idx_dict[idx]
            
        return score



from text_chunk import TextChunker
class BM25Retriever:
    """BM25アルゴリズムを使用してドキュメントを検索するためのリトリーバークラス"""

    def __init__(self, chunk_size: int = 768, chunk_overlap: int = 12):
        """BM25Retrieverを初期化します。

        Args:
            chunk_size (int, optional): テキストチャンクのサイズ。デフォルトは768。
            chunk_overlap (int, optional): チャンク間の重複文字数。デフォルトは12。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # TextEmbedderの代わりにBM25TextEmbeddingを使用
        self.embedder = BM25TextEmbedding()
        self.documents = []
        self.chunks = []
        self.embeddings = []
        self.chunk_to_doc_map = {}  # チャンクとドキュメントのマッピング

    def create_query_vector(self, query: str, **kwargs) -> Dict[int, float]:
        """
        クエリテキストから検索ベクトルを作成

        Args:
            query: クエリテキスト
            **kwargs: 追加のオプション

        Returns:
            Dict[int, float]: 検索ベクトル (インデックスとスコアのマップ)
        """
        # クエリの埋め込みを取得
        query_embedding = next(self.embedder.query_embed(query))
        
        # 辞書形式に変換
        query_vector = {}
        for idx, val in zip(query_embedding.indices, query_embedding.values):
            query_vector[idx] = val
            
        return query_vector

    def create_passage_vector(self, text: str, **kwargs) -> Dict[int, float]:
        """
        文書テキストから文書ベクトルを作成

        Args:
            text: 文書テキスト
            **kwargs: 追加のオプション

        Returns:
            Dict[int, float]: 文書ベクトル (インデックスとスコアのマップ)
        """
        # 文書の埋め込みを取得
        # 単一の文書を埋め込むためにリストにして処理
        embedding = next(self.embedder.embed(text))
        
        # 辞書形式に変換
        document_vector = {}
        for idx, val in zip(embedding.indices, embedding.values):
            document_vector[idx] = val
            
        return document_vector

    def add_documents(self, documents: List[str]) -> None:
        """ドキュメントをインデックスに追加します。

        Args:
            documents (List[str]): 追加するドキュメントのリスト
        """
        start_idx = len(self.documents)
        self.documents.extend(documents)
        
        # ドキュメントをチャンクに分割し、各チャンクの元のドキュメントを追跡
        all_chunks = []
        chunk_to_doc_idx = {}
        
        for doc_idx, doc in enumerate(documents, start=start_idx):
            doc_chunks = self.chunker.split_text(doc)
            for chunk_idx in range(len(all_chunks), len(all_chunks) + len(doc_chunks)):
                chunk_to_doc_idx[chunk_idx] = doc_idx
            all_chunks.extend(doc_chunks)
        
        # 既存のチャンク数
        existing_chunks_count = len(self.chunks)
        
        # チャンクとドキュメントのマッピングを更新（既存のチャンク数を考慮）
        for chunk_idx, doc_idx in chunk_to_doc_idx.items():
            self.chunk_to_doc_map[existing_chunks_count + chunk_idx] = doc_idx
        
        # チャンクの埋め込みを生成
        embeddings = list(self.embedder.embed(all_chunks))
        
        # チャンクと埋め込みを保存
        self.chunks.extend(all_chunks)
        self.embeddings.extend(embeddings)

    def search(self, query: str, passages: List[str] = None, top_k: int = 5, **kwargs) -> List[Tuple[str, float]]:
        """クエリに最も類似したドキュメントを検索します。

        Args:
            query (str): 検索クエリ
            passages: 検索対象の文書リスト（Noneの場合はインデックス済み文書を使用）
            top_k (int, optional): 返す結果の最大数。デフォルトは5。
            **kwargs: その他のパラメータ

        Returns:
            List[Tuple[str, float]]: (ドキュメント, スコア)のタプルのリスト
        """
        # passagesが指定されていない場合、インデックス済みのドキュメントを使用
        if passages is None:
            if not self.embeddings:
                return []
                
            # クエリの埋め込みを生成
            query_embedding = next(self.embedder.query_embed(query))
            
            # 類似度スコアを計算
            scores = []
            for i, embedding in enumerate(self.embeddings):
                # スコアを計算
                score = self.embedder.calculate_similarity(query_embedding, embedding)
                scores.append((i, score))
            
            # スコアで降順ソート
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # 上位のチャンクを取得
            top_chunk_indices = [idx for idx, _ in scores[:top_k]]
            
            # チャンクをドキュメントにマッピング
            results = []
            seen_doc_indices = set()
            
            for chunk_idx in top_chunk_indices:
                doc_idx = self.chunk_to_doc_map.get(chunk_idx, 0)
                if doc_idx not in seen_doc_indices:
                    seen_doc_indices.add(doc_idx)
                    if 0 <= doc_idx < len(self.documents):
                        score = scores[top_chunk_indices.index(chunk_idx)][1]
                        results.append((self.documents[doc_idx], score))
            
            # 上位k件のドキュメントを返す（既に同じドキュメントからのチャンクは除外されている）
            return results[:top_k]
        else:
            # 新しいパッセージセットに対して検索
            # 各文書の類似度を計算
            results = []
            for passage in passages:
                # 文書ベクトルを作成
                passage_vector = self.create_passage_vector(passage)
                
                # クエリベクトルを作成
                query_vector = self.create_query_vector(query)
                
                # 類似度スコアを計算
                score = 0.0
                for token_id, query_weight in query_vector.items():
                    if token_id in passage_vector:
                        score += query_weight * passage_vector[token_id]
                        
                results.append((passage, score))
            
            # スコアの降順でソート
            results.sort(key=lambda x: x[1], reverse=True)
            
            # 上位k件を返す
            return results[:top_k]

    # 下位互換性のために残しておく
    def similarity_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """クエリに最も類似したドキュメントを検索します。

        Args:
            query (str): 検索クエリ
            top_k (int, optional): 返す結果の最大数。デフォルトは5。

        Returns:
            List[Tuple[str, float]]: (ドキュメント, スコア)のタプルのリスト
        """
        return self.search(query, top_k=top_k)


if __name__ == "__main__":
    # BM25TextEmbeddingとBM25Retrieverのテスト
    import time

    # テスト用のサンプルテキスト
    sample_documents = [
        "東京は日本の首都で、人口が最も多い都市です。多くの観光名所や企業の本社があります。",
        "大阪は日本第二の都市で、関西の中心地です。食文化が豊かで「天下の台所」と呼ばれています。",
        "札幌は北海道の中心都市で、冬季オリンピックが開催されたこともある雪の街です。",
        "福岡は九州の中心都市で、アジア諸国との交流も盛んです。",
        "京都は古都と呼ばれ、多くの歴史的建造物が残っています。観光地として人気があります。"
    ]

    print("=== BM25TextEmbedding テスト ===")
    # BM25TextEmbedding インスタンス作成
    embedder = BM25TextEmbedding()
    
    # テキストの埋め込みを生成
    start_time = time.time()
    embeddings = list(embedder.embed(sample_documents))
    print(f"埋め込み生成時間: {time.time() - start_time:.4f}秒")
    print(f"埋め込みの数: {len(embeddings)}")
    
    # 埋め込みの形状を確認
    for i, embedding in enumerate(embeddings):
        print(f"embedding {i}: インデックス数={len(embedding.indices)}, 値数={len(embedding.values)}")
    
    # クエリの埋め込みを生成
    query = "日本の観光地について教えてください"
    query_embedding = next(embedder.query_embed(query))
    print(f"\nクエリ: \"{query}\"")
    print(f"クエリ埋め込み: インデックス数={len(query_embedding.indices)}, 値数={len(query_embedding.values)}")
    
    # 類似度計算のテスト
    print("\n=== 類似度スコア ===")
    for i, embedding in enumerate(embeddings):
        similarity = embedder.calculate_similarity(query_embedding, embedding)
        print(f"ドキュメント {i}: {similarity:.4f}")
    
    print("\n=== BM25Retriever テスト ===")
    # BM25Retrieverのインスタンス作成
    retriever = BM25Retriever(chunk_size=200, chunk_overlap=20)
    
    # ドキュメントの追加
    retriever.add_documents(sample_documents)
    print(f"インデックス済みドキュメント数: {len(retriever.documents)}")
    print(f"チャンク数: {len(retriever.chunks)}")
    
    # 検索テスト
    queries = [
        "首都について教えてください",
        "観光地はどこですか",
        "食べ物が有名な都市はどこですか"
    ]
    
    for test_query in queries:
        print(f"\nクエリ: \"{test_query}\"")
        results = retriever.search(test_query, top_k=3)
        
        for i, (doc, score) in enumerate(results):
            print(f"結果 {i+1} (スコア: {score:.4f}):")
            print(f"  {doc[:100]}...")