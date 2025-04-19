from sentence_transformers import SentenceTransformer
import torch
import mmh3
import math
import stopwordsiso
import numpy as np
from sudachipy import tokenizer
from sudachipy import dictionary
from typing import List, Dict, Optional, Union, Iterable, Any
from fastembed import SparseEmbedding


class BM42TextEmbedding:
    """BM42アルゴリズムを使用したテキスト埋め込みクラス"""

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-small-v2",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        alpha: float = 0.5
    ):
        """BM42TextEmbeddingを初期化します。

        Args:
            model_name: 使用するSentenceTransformerモデルの名前
            device: モデルを実行するデバイス（'cuda'または'cpu'）
            alpha: BM42スコアリングの重み付けパラメータ
        """

        #
        self._tokenizer = dictionary.Dictionary().create()
        self._tokenizer_mode = tokenizer.Tokenizer.SplitMode.B
        self._stopwords = stopwordsiso.stopwords("ja")

        self.model_name = model_name
        self.device = device
        self.alpha = alpha
        self.model = None
        self.load_model(model_name, device)

    def load_model(self, model_name: str, device: str):
        """SentenceTransformerモデルをロードします。

        Args:
            model_name: 使用するSentenceTransformerモデルの名前
            device: モデルを実行するデバイス（'cuda'または'cpu'）
        """
        self.model = SentenceTransformer(model_name, trust_remote_code=True, device=device)

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

    def _token_attentions(self,
                          text: str,
                          is_query: bool = False
                          ) -> Dict[str, float]:
        """モデルからトークンの注目度（Attention）を取得します。Sudachiが有効な場合は形態素単位で結合します。

        Args:
            text: 分析するテキスト
            is_query: テキストをクエリとして扱うかどうか（接頭辞に影響）

        Returns:
            Dict[str, float]: トークンとその注目度を含む辞書
        """
        # モデル入力用に接頭辞を追加
        if is_query:
            if not text.startswith("クエリ: "):
                prefixed_text = f"クエリ: {text}"
            else:
                prefixed_text = text
        else:
            if not text.startswith("文章: "):
                prefixed_text = f"文章: {text}"
            else:
                prefixed_text = text

        # SentenceTransformerの内部モデルとtokenizerにアクセス
        transformer_model = self.model[0].auto_model
        tokenizer = self.model.tokenizer
        device = self.model.device  # モデルのデバイスを取得

        # テキストをトークン化
        inputs = tokenizer(prefixed_text, return_tensors="pt", truncation=True, max_length=512)

        # 入力テンソルをモデルと同じデバイスに移動
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 注目度（Attention）を含む出力を取得
        with torch.no_grad():
            outputs = transformer_model(**inputs, output_attentions=True)

        # 最後の層の注目度を取得し、[CLS]トークン（0番目）に対する平均注目度を計算
        attentions = outputs.attentions[-1][0, :, 0].mean(dim=0)

        # CPUに移動して処理
        attentions = attentions.cpu()
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())

        # [CLS]を除いたトークンと重み
        raw_tokens = [(t, float(w)) for t, w in zip(tokens[1:-1], attentions[1:-1])]

        # トークンを処理（特殊記号を除去）して重みとペアに
        token_pairs = []
        for token, weight in raw_tokens:
            # SentencePieceの▁（U+2581）を除去
            clean_token = token.replace('##', '').replace('▁', '').strip()
            if clean_token:
                token_pairs.append((clean_token, weight))

        # 接頭辞のスキップ（ruri-small-v2観測なので固有）
        if is_query:
            # クエリの場合：「ク」「エ」「リ」「:」の4トークンをスキップする可能性がある
            if len(token_pairs) >= 4:
                # 最初の3トークンが「ク」「エ」「リ」かチェック
                if (token_pairs[0][0] == 'ク' and
                        token_pairs[1][0] == 'エ' and
                        token_pairs[2][0] == 'リ' and
                        token_pairs[3][0] == ':'):
                    # 接頭辞の4トークンをスキップ
                    # 場合によっては次の空白トークンもスキップ
                    skip_count = 4
                    if len(token_pairs) > 4 and not token_pairs[4][0]:  # 空のトークンをチェック
                        skip_count = 5
                    processed_tokens = token_pairs[skip_count:]
                else:
                    processed_tokens = token_pairs
            else:
                processed_tokens = token_pairs
        else:
            # 文章の場合：「文章」「:」の2トークンをスキップする可能性がある
            if len(token_pairs) >= 2:
                # 最初のトークンが「文章」か、2番目が「:」かチェック
                if token_pairs[0][0] == '文章' and token_pairs[1][0] == ':':
                    # 接頭辞の2トークンをスキップ
                    # 場合によっては次の空白トークンもスキップ
                    skip_count = 2
                    if len(token_pairs) > 2 and not token_pairs[2][0]:  # 空のトークンをチェック
                        skip_count = 3
                    processed_tokens = token_pairs[skip_count:]
                else:
                    processed_tokens = token_pairs
            else:
                processed_tokens = token_pairs

        base_token_attentions = {}
        for token, weight in processed_tokens:
            # 同じトークンがすでに辞書にある場合は、平均値を取る
            if token in base_token_attentions:
                base_token_attentions[token] = (base_token_attentions[token] + weight) / 2
            else:
                base_token_attentions[token] = weight

        # 単語の表層形を取得
        ja_tokens = self._tokenize(text)

        result = {}
        for token in ja_tokens:
            result[token] = 0
            for t, weight in base_token_attentions.items():
                if t in token:
                    result[token] += weight

        return result

    def _rescore_token_weights(self, token_weights: Dict[str, float]) -> Dict[int, float]:
        """BM42に基づいてトークンの重みを再計算します。

        Args:
            token_weights: トークンとその重みの辞書

        Returns:
            Dict[int, float]: ハッシュされたトークンIDと再計算された重みの辞書
        """
        rescored = {}

        for token, value in token_weights.items():
            if value <= 0.:
                continue
            token_id = abs(mmh3.hash(token))
            # Log(1/rank + 1)^alpha の計算式でスコアを調整
            rescored[token_id] = math.log(1.0 + value) ** self.alpha

        return rescored

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """ドキュメントに対するBM42埋め込みを生成します。

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
        
        # 文書ごとに埋め込みを生成
        for i, doc in enumerate(documents):
            # 文書ベクトルを生成
            token_attentions = self._token_attentions(
                doc,
                is_query=False,
            )
            passage_vector = self._rescore_token_weights(token_attentions)
            indices = np.array(list(passage_vector.keys()), dtype=np.int32)
            values = np.array(list(passage_vector.values()), dtype=np.float32)

            # スパース埋め込みを作成
            embedding = SparseEmbedding(indices=indices, values=values)
            yield embedding

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        """クエリに対するBM42埋め込みを生成します。

        Args:
            query: 埋め込みを生成するクエリテキストまたはクエリのリスト
            **kwargs: 追加のパラメータ

        Returns:
            Iterable[SparseEmbedding]: クエリの埋め込み
        """
        return self.embed(query)
