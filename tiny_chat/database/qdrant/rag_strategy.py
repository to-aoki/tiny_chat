
from abc import ABC
from qdrant_client import models

from tiny_chat.database.embeddings.bm25_embedding import BM25TextEmbedding
from tiny_chat.database.embeddings.static_embedding import StaticEmbedding


class RagStrategyFactory:

    @staticmethod
    def get_strategy(strategy_name, use_gpu=False) -> 'RAGStrategy':
        strategy_name = strategy_name.lower()
        if strategy_name == 'bm25':
            return SparseOnly('bm25')
        elif strategy_name == 'bm42':
            return SparseOnly('bm42', use_gpu=use_gpu)
        elif strategy_name == 'ruri_small':
            return DenseOnly("cl-nagoya/ruri-small-v2", use_gpu=use_gpu)
        elif strategy_name == 'ja_static':
            return DenseOnly("hotchpotch/static-embedding-japanese", use_gpu=use_gpu)
        elif strategy_name == 'bm25_static':
            return SparceDenseRRF('bm25_static', use_gpu=use_gpu)
        elif strategy_name == 'bm25_sbert':
            return SparceDenseRRF('bm25_sbert', use_gpu=use_gpu)
        return None


class RAGStrategy(ABC):

    def use_vector_name(self):
        return None

    def create_vector_config(self):
        return None

    def create_sparse_vectors_config(self):
        return None

    def vector(self, text):
        return {}

    def prefetch(self, text):
        return []

    def query(self, text=None):
        return None


class SparseOnly(RAGStrategy):

    def __init__(self, strategy='bm25', use_gpu=False):
        self.strategy = strategy

        self.sparse_vector_field_name = "sparse"
        if strategy == 'bm25':
            self.model = BM25TextEmbedding()
        elif strategy == 'bm42':
            from tiny_chat.database.embeddings.bm42_embedding import BM42TextEmbedding
            self.model = BM42TextEmbedding(device='cuda' if use_gpu else 'cpu')
        else:
            raise ValueError('unknown Strategy: ' + strategy)

    def create_vector_config(self):
        return {}

    def use_vector_name(self):
        return self.sparse_vector_field_name

    def create_sparse_vectors_config(self):
        return {
            self.sparse_vector_field_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }

    def vector(self, text):
        sparse_embedding = list(self.model.embed(text))[0]
        return {
            self.sparse_vector_field_name: sparse_embedding.as_object(),
        }

    def prefetch(self, text, top_k):
        return []

    def query(self, text=None):
        query_embedding = list(self.model.query_embed(text))[0]
        return models.SparseVector(
            values=query_embedding.values.tolist(),
            indices=query_embedding.indices.tolist()
        )


class DenseOnly(RAGStrategy):

    def __init__(self, strategy="cl-nagoya/ruri-small-v2", use_gpu=False):

        self.strategy = strategy
        self.dense_vector_field_name = "dense"
        if strategy == "hotchpotch/static-embedding-japanese":
            self.model = StaticEmbedding(
                model_name=strategy, device='cuda' if use_gpu else 'cpu')
        else:
            from tiny_chat.database.embeddings.stransformer_embedding import SentenceTransformerEmbedding
            self.model = SentenceTransformerEmbedding(
                model_name=strategy, device='cuda' if use_gpu else 'cpu')

    def create_vector_config(self):
        return {
            self.dense_vector_field_name: models.VectorParams(
                size=self.model.dimension,
                distance=models.Distance.COSINE,
            )
        }

    def use_vector_name(self):
        return self.dense_vector_field_name

    def create_sparse_vectors_config(self):
        return {}

    def vector(self, text):
        dense_embedding = list(self.model.embed(text))[0]
        return {
            self.dense_vector_field_name: dense_embedding.tolist()
        }

    def prefetch(self, text, top_k):
        return []

    def query(self, text=None):
        query_embedding = list(self.model.query_embed(text))[0]
        return query_embedding.tolist()


class SparceDenseRRF(RAGStrategy):

    def __init__(self, strategy='bm25_static', use_gpu=False):
        self.strategy = strategy
        self.sparse_vector_field_name = "sparse"
        self.dense_vector_field_name = "dense"
        if self.strategy == 'bm25_static':
            self.bm25_model = BM25TextEmbedding()
            self.emb_model = StaticEmbedding(device='cuda' if use_gpu else 'cpu')
        elif self.strategy == 'bm25_sbert':
            self.bm25_model = BM25TextEmbedding()
            from tiny_chat.database.embeddings.stransformer_embedding import SentenceTransformerEmbedding
            self.emb_model = SentenceTransformerEmbedding(device='cuda' if use_gpu else 'cpu')

    def create_vector_config(self):
        return {
            self.dense_vector_field_name: models.VectorParams(
                size=self.emb_model.dimension,
                distance=models.Distance.COSINE,
            )
        }

    def create_sparse_vectors_config(self):
        return {
            self.sparse_vector_field_name: models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        }

    def vector(self, text):
        sparse_embedding = list(self.bm25_model.embed(text))[0]
        dense_embedding = list(self.emb_model.embed(text))[0]
        return {
            self.dense_vector_field_name: dense_embedding.tolist(),
            self.sparse_vector_field_name: sparse_embedding.as_object(),
        }

    def prefetch(self, text, top_k):
        sparse_embedding = list(self.bm25_model.query_embed(text))[0]
        dense_embedding = list(self.emb_model.query_embed(text))[0]
        return [
            models.Prefetch(
                query=sparse_embedding.as_object(), using=self.sparse_vector_field_name, limit=top_k),
            models.Prefetch(
                query=dense_embedding.tolist(), using=self.dense_vector_field_name, limit=top_k),
        ]

    def query(self, text=None):
        return models.FusionQuery(fusion=models.Fusion.RRF)