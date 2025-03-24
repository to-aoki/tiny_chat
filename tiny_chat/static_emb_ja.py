from typing import Any, Iterable, Optional, Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from fastembed.common.types import NumpyArray


class StaticEmbeddingJapanese:
    """
    A text embedding class using a Japanese static embedding model.
    This class wraps the SentenceTransformer model to provide a consistent interface
    for Japanese text embeddings.
    """

    def __init__(
        self,
        model_name="hotchpotch/static-embedding-japanese",
        device="cpu",
        **kwargs
    ):
        """
        Initialize the StaticEmbeddingJapanese with a specific model.

        Args:
            model_name (str): Name of the model to use.
            device (str): Device to use for computation ('cpu' or 'cuda').
            **kwargs: Additional arguments to pass to SentenceTransformer.
        """
        self.model = SentenceTransformer(model_name, device=device, **kwargs)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def _ensure_list(self, documents: Union[str, Iterable[str]]) -> list[str]:
        """Convert input to list if it's a single string."""
        if isinstance(documents, str):
            return [documents]
        return list(documents)

    def _batch_process(
        self, 
        documents: list[str], 
        batch_size: int = 256
    ) -> list[NumpyArray]:
        """Process documents in batches to save memory."""
        results = []
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            embeddings = self.model.encode(
                batch, 
                convert_to_tensor=False, 
                show_progress_bar=False
            )
            results.extend(embeddings)
        return results

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Encode a list of documents into list of embeddings.
        We use the SentenceTransformer model to handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default threading instead.
            **kwargs: Additional keyword arguments to pass to the encode method.

        Yields:
            Embeddings, one per document
        """
        doc_list = self._ensure_list(documents)
        
        # Process in batches
        embeddings = self._batch_process(doc_list, batch_size)

        # Yield the results
        for embedding in embeddings:
            yield embedding

    def query_embed(self, query: Union[str, Iterable[str]], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds queries using the SentenceTransformer model.

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.
            **kwargs: Additional keyword arguments to pass to the embed method.

        Yields:
            Iterable[NumpyArray]: The embeddings.
        """
        # For this model, query embedding is the same as document embedding
        yield from self.embed(query, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword arguments to pass to the embed method.

        Yields:
            Iterable[NumpyArray]: The dense embeddings.
        """
        # For this model, passage embedding is the same as document embedding
        yield from self.embed(texts, **kwargs)

    def similarity(self, embeddings1: NumpyArray, embeddings2: NumpyArray) -> NumpyArray:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embeddings1 (NumpyArray): First set of embeddings.
            embeddings2 (NumpyArray): Second set of embeddings.

        Returns:
            NumpyArray: Cosine similarity matrix.
        """
        return cos_sim(embeddings1, embeddings2)


if __name__ == "__main__":
    import time
    
    # テストデータ
    query = "美味しいラーメン屋に行きたい"
    documents = [
        "素敵なカフェが近所にあるよ。落ち着いた雰囲気でゆっくりできるし、窓際の席からは公園の景色も見えるんだ。",
        "新鮮な魚介を提供する店です。地元の漁師から直接仕入れているので鮮度は抜群ですし、料理人の腕も確かです。",
        "あそこは行きにくいけど、隠れた豚骨の名店だよ。スープが最高だし、麺の硬さも好み。",
        "おすすめの中華そばの店を教えてあげる。とりわけチャーシューが手作りで柔らかくてジューシーなんだ。",
        "駅前にある古い建物の中のイタリアンレストランは、本格的なパスタが食べられると評判です。",
    ]
    
    print("=== StaticEmbeddingJapanese のテスト ===")
    
    # 1. クラスの初期化
    print("\n1. クラスの初期化")
    start_time = time.time()
    embedding = StaticEmbeddingJapanese()
    print(f"初期化時間: {time.time() - start_time:.4f}秒")
    print(f"埋め込みの次元数: {embedding.dimension}")
    
    # 2. クエリの埋め込み
    print("\n2. クエリの埋め込み (query_embed)")
    start_time = time.time()
    query_embedding = list(embedding.query_embed(query))[0]
    print(f"クエリの埋め込み時間: {time.time() - start_time:.4f}秒")
    print(f"クエリの埋め込みの形状: {query_embedding.shape}")
    
    # 3. ドキュメントの埋め込み
    print("\n3. ドキュメントの埋め込み (passage_embed)")
    start_time = time.time()
    document_embeddings = list(embedding.passage_embed(documents))
    print(f"ドキュメントの埋め込み時間: {time.time() - start_time:.4f}秒")
    print(f"ドキュメント数: {len(document_embeddings)}")
    print(f"各ドキュメントの埋め込みの形状: {document_embeddings[0].shape}")
    
    # 4. 類似度の計算
    print("\n4. 類似度の計算")
    similarities = embedding.similarity(query_embedding, document_embeddings)
    
    # 5. 結果の表示
    print("\n5. 結果の表示（類似度順）")
    similarity_scores = similarities[0].tolist()
    ranked_results = sorted(zip(similarity_scores, documents), reverse=True)
    
    for i, (score, doc) in enumerate(ranked_results):
        print(f"{i+1}. スコア: {score:.4f} - {doc[:50]}{'...' if len(doc) > 50 else ''}")
    
    # 6. バッチ処理のテスト
    print("\n6. バッチ処理のテスト")
    start_time = time.time()
    batch_size = 2
    batch_embeddings = list(embedding.embed(documents, batch_size=batch_size))
    print(f"バッチサイズ {batch_size} でのエンコード時間: {time.time() - start_time:.4f}秒")
    print(f"埋め込み数: {len(batch_embeddings)}")
    
    print("\n=== テスト完了 ===")

