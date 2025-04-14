import os.path
from typing import List, Union, Iterable, Optional, Any, Sequence

from sudachipy import tokenizer
from sudachipy import dictionary
import stopwordsiso
from fastembed import SparseEmbedding, SparseTextEmbedding
from fastembed.common import OnnxProvider

from fastembed.common.model_description import SparseModelDescription, ModelSource
from fastembed.sparse.bm25 import supported_bm25_models


# qdrant bm25は日本語以外のSTEM辞書が配布される。日本語は未対応で自前でやるのでダウンロードは不要
supported_bm25_models.append(
    SparseModelDescription(
        model="bm25-ja",
        vocab_size=0,
        description="BM25 japanese model (only support japanese)",
        license="MIT",
        sources=ModelSource(url="dummy"),
        size_in_GB=0.01,
        requires_idf=True,
        model_file="mock.file",
    )
)


class BM25TextEmbedding(SparseTextEmbedding):

    def __init__(
        self,
        model_name: str = "bm25-ja",
        cache_dir: Optional[str] = os.path.dirname(__file__) + "/model",
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

        # 内部的には英語モードとなり表層は日本語利用
        self.is_japanese = True
        if language != "japanese":
            kwargs["language"] = language
            self.is_japanese = False

        kwargs["k"] = k
        kwargs["b"] = b
        kwargs["avg_len"] = avg_len
        kwargs["token_max_length"] = token_max_length
        kwargs["local_files_only"] = True
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
            self._tokenizer_mode = tokenizer.Tokenizer.SplitMode.B
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
            return [morph.normalized_form() for morph in morphemes]
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

