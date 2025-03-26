from typing import List, Dict, Tuple, Union, Iterable, Optional, Any
import math
from collections import Counter, defaultdict

from sudachipy import tokenizer
from sudachipy import dictionary
from fastembed import SparseEmbedding
import stopwordsiso


class BM25TextEmbedding:
    """BM25アルゴリズムを使用したテキスト埋め込みクラス"""

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        avg_doc_length: float = 40,
    ):
        """BM25TextEmbeddingを初期化します。

        Args:
            k1 (float): BM25のk1パラメータ (デフォルト: 1.5)
            b (float): BM25のbパラメータ (デフォルト: 0.75)
        """
        # BM25パラメータ
        self.k1 = k1
        self.b = b
        
        # 語彙とIDF値を格納する辞書
        self.vocabulary = {}  # token -> token_id のマッピング
        self.idf = {}         # token_id -> idf値 のマッピング
        self.avg_doc_length = avg_doc_length
        self.doc_count = 0
        self.token_count = 0  # ボキャブラリのサイズ
        
        # コーパス統計
        self.doc_lengths = []  # 各文書のトークン数
        self.doc_freqs = defaultdict(int)  # 各トークンが出現する文書数
        
        # 日本語処理のためのコンポーネント
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

    def _get_or_add_token_id(self, token: str) -> int:
        """トークンのIDを取得または新規追加します。

        Args:
            token (str): トークン

        Returns:
            int: トークンID
        """
        if token not in self.vocabulary:
            self.vocabulary[token] = self.token_count
            self.token_count += 1
        return self.vocabulary[token]

    def _calculate_idf(self) -> None:
        """トークンのIDF値を計算します。"""
        for token_id, doc_freq in self.doc_freqs.items():
            # BM25のIDF計算式
            self.idf[token_id] = math.log((self.doc_count - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)

    def _process_documents(self, documents: List[str]) -> List[List[int]]:
        """ドキュメントを処理してトークンIDのリストに変換します。

        Args:
            documents (List[str]): 処理するドキュメントのリスト

        Returns:
            List[List[int]]: 各ドキュメントのトークンIDのリスト
        """
        tokenized_docs = []
        total_length = 0
        
        for doc in documents:
            # テキストをトークン化
            tokens = self._tokenize(doc)
            
            # トークンをIDに変換
            token_ids = [self._get_or_add_token_id(token) for token in tokens]
            tokenized_docs.append(token_ids)
            
            # 文書の長さを記録
            doc_length = len(token_ids)
            self.doc_lengths.append(doc_length)
            total_length += doc_length
            
            # 文書中の各トークンの出現を記録
            unique_tokens = set(token_ids)
            for token_id in unique_tokens:
                self.doc_freqs[token_id] += 1
        
        # ドキュメント数と平均長を更新
        self.doc_count += len(documents)
        if self.doc_count > 0:
            self.avg_doc_length = total_length / self.doc_count
        
        # IDFを計算
        self._calculate_idf()
        
        return tokenized_docs

    def _bm25_score(self, term_freqs: Dict[int, int], doc_index: int) -> Dict[int, float]:
        """BM25スコアリングを実行します。

        Args:
            term_freqs (Dict[int, int]): ドキュメント内の各トークンの出現頻度
            doc_index (int): ドキュメントのインデックス

        Returns:
            Dict[int, float]: トークンIDとそのBM25スコアのマッピング
        """
        scores = {}
        doc_len = self.doc_lengths[doc_index]
        
        for token_id, tf in term_freqs.items():
            if token_id in self.idf:
                # BM25スコア計算式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
                scores[token_id] = self.idf[token_id] * (numerator / denominator)
        
        return scores

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """ドキュメントに対するBM25埋め込みを生成します。

        Args:
            documents: 埋め込みを生成するテキストまたはテキストのリスト
            batch_size: エンコーディングのバッチサイズ
            parallel: 並列処理の使用（0: 全コア使用、None: 使用しない）
            **kwargs: 追加のパラメータ

        Returns:
            Iterable[SparseEmbedding]: 埋め込みのシーケンス
        """
        # 単一文書の場合リストに変換
        if isinstance(documents, str):
            documents = [documents]
        
        # 文書を処理
        tokenized_docs = self._process_documents(documents)
        
        # 文書ごとに埋め込みを生成
        embeddings = []
        
        for i, token_ids in enumerate(tokenized_docs):
            # 文書内の各トークンの出現頻度を計算
            term_freqs = Counter(token_ids)
            
            # BM25スコアを計算
            scores = self._bm25_score(term_freqs, i)
            
            # 非ゼロのスコアを持つトークンのみをスパース埋め込みとして抽出
            indices = []
            values = []
            
            for token_id, score in scores.items():
                indices.append(token_id)
                values.append(score)
            
            # スパース埋め込みを作成
            embedding = SparseEmbedding(indices=indices, values=values)
            embeddings.append(embedding)
            
            # イテレータとして返す
            yield embedding

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        """クエリに対するBM25埋め込みを生成します。

        Args:
            query: 埋め込みを生成するクエリテキストまたはクエリのリスト
            **kwargs: 追加のパラメータ

        Returns:
            Iterable[SparseEmbedding]: クエリの埋め込み
        """
        # 単一クエリの場合
        if isinstance(query, str):
            tokens = self._tokenize(query)
            token_ids = [self._get_or_add_token_id(token) for token in tokens]
            
            # 各トークンのTF-IDFスコアを計算
            indices = []
            values = []
            
            # クエリトークンの出現頻度をカウント
            term_freqs = Counter(token_ids)
            
            for token_id, tf in term_freqs.items():
                if token_id in self.idf:
                    indices.append(token_id)
                    # クエリの場合は単純なTF-IDFを使用（BM25より単純化）
                    values.append(tf * self.idf[token_id])
            
            # スパース埋め込みを作成
            embedding = SparseEmbedding(indices=indices, values=values)
            yield embedding
        else:
            # 複数クエリの場合
            for q in query:
                tokens = self._tokenize(q)
                token_ids = [self._get_or_add_token_id(token) for token in tokens]
                
                indices = []
                values = []
                term_freqs = Counter(token_ids)
                
                for token_id, tf in term_freqs.items():
                    if token_id in self.idf:
                        indices.append(token_id)
                        values.append(tf * self.idf[token_id])
                
                embedding = SparseEmbedding(indices=indices, values=values)
                yield embedding

    def calculate_similarity(self, query_embedding: SparseEmbedding, document_embedding: SparseEmbedding) -> float:
        """クエリ埋め込みとドキュメント埋め込み間の類似度を計算します。

        Args:
            query_embedding (SparseEmbedding): クエリの埋め込み
            document_embedding (SparseEmbedding): ドキュメントの埋め込み

        Returns:
            float: 類似度スコア
        """
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
        # 独自のBM25TextEmbeddingを使用
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
        print(embedding.values)
    
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