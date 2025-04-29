import os.path
from typing import Any, Iterable, Optional, Union
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from fastembed.common.types import NumpyArray


class SentenceTransformerEmbedding:

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-30m",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_query_prefix = True,
        **kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.use_query_prefix = use_query_prefix
        extra_path = kwargs.get("file_name", None)
        if device == 'cpu' and extra_path is not None and extra_path.startswith("openvino/"):
            model_dir = os.path.dirname(__file__) + "/model/" + os.path.split(model_name)[-1]
            if not os.path.isdir(model_dir):
                # pip install sentence-transformers[openvino]
                print("モデル変換(openvino int8) 開始:", model_dir)
                model = SentenceTransformer(
                    model_name,
                    trust_remote_code=True,
                    backend="openvino"
                )
                from sentence_transformers import export_static_quantized_openvino_model
                from optimum.intel import OVQuantizationConfig
                model.save(model_dir)
                quantization_config = OVQuantizationConfig()
                export_static_quantized_openvino_model(model, quantization_config, model_dir)
                print("モデル変換(openvino int8) 完了:", model_dir)

            self.model = SentenceTransformer(
                model_dir,
                device=device,
                backend="openvino",
                model_kwargs=kwargs
            )
        else:
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device,
                model_kwargs=kwargs
            )

        if device.startswith('cuda'):
            self.model.half()
        self.dimension = self.model.get_sentence_embedding_dimension()

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
        is_query: bool = False,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        if isinstance(documents, str):
            documents = [documents]

        if is_query:
            prefix = "検索クエリ: "
        else:
            prefix = "検索文章: "

        for i, doc in enumerate(documents):
            if not doc.startswith(prefix):
                documents[i] = f"{prefix}{doc}"
            else:
                documents[i] = doc

        doc_list = list(documents)
        embeddings = self._batch_process(doc_list, batch_size)

        # Yield the results
        for embedding in embeddings:
            yield embedding

    def query_embed(
        self, query: Union[str, Iterable[str]], *kwargs: Any
    ) -> Iterable[NumpyArray]:
        # For this model, query embedding is the same as document embedding
        yield from self.embed(query, is_query=self.use_query_prefix, **kwargs)

    def similarity(self, embeddings1: NumpyArray, embeddings2: NumpyArray) -> NumpyArray:
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

    print("=== SentenceTransformerEmbedding のテスト ===")

    # 1. クラスの初期化
    print("\n1. クラスの初期化")
    start_time = time.time()
    embedding = SentenceTransformerEmbedding()
    print(f"初期化時間: {time.time() - start_time:.4f}秒")
    print(f"埋め込みの次元数: {embedding.dimension}")

    # 2. クエリの埋め込み
    print("\n2. クエリの埋め込み (query_embed)")
    start_time = time.time()
    query_embedding = list(embedding.query_embed(query))[0]
    print(f"クエリの埋め込み時間: {time.time() - start_time:.4f}秒")
    print(f"クエリの埋め込みの形状: {query_embedding.shape}")

    # 3. ドキュメントの埋め込み
    print("\n3. ドキュメントの埋め込み (embed)")
    start_time = time.time()
    document_embeddings = list(embedding.embed(documents))
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
        print(f"{i + 1}. スコア: {score:.4f} - {doc[:50]}{'...' if len(doc) > 50 else ''}")

    # 6. バッチ処理のテスト
    print("\n6. バッチ処理のテスト")
    start_time = time.time()
    batch_size = 2
    batch_embeddings = list(embedding.embed(documents, batch_size=batch_size))
    print(f"バッチサイズ {batch_size} でのエンコード時間: {time.time() - start_time:.4f}秒")
    print(f"埋め込み数: {len(batch_embeddings)}")

    print("\n=== テスト完了 ===")
