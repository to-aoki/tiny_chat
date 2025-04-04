
from abc import ABC
from qdrant_client import models

from tiny_chat.database.embeddings.bm25_embedding import BM25TextEmbedding
from tiny_chat.database.embeddings.static_embedding import StaticEmbedding


class RagStrategyFactory:

    @staticmethod
    def get_strategy(strategy_name, use_gpu=False) -> 'RAGStrategy':
        strategy_name = strategy_name.lower()
        if strategy_name == "bm25":
            return SparseOnly("bm25")
        elif strategy_name == "bm42":
            return SparseOnly("bm4", use_gpu=use_gpu)
        elif strategy_name == "splide_ja":
            return SparseOnly("hotchpotch/japanese-splade-v2", use_gpu=use_gpu)
        elif strategy_name == "ruri_small":
            return DenseOnly("cl-nagoya/ruri-small-v2", use_gpu=use_gpu)
        elif strategy_name == "ja_static":
            return DenseOnly("hotchpotch/static-embedding-japanese", use_gpu=use_gpu)
        elif strategy_name == "m_e5":
            return DenseOnly("intfloat/multilingual-e5-large", use_gpu=use_gpu)
        elif strategy_name == "bm25_static":
            return SpaceDenseRRF("bm25_static", use_gpu=use_gpu)
        elif strategy_name == "bm25_sbert":
            return SpaceDenseRRF("bm25_sbert", use_gpu=use_gpu)
        elif strategy_name == "bm25_splide":
            return SpaceRRF("bm25_splide", use_gpu=use_gpu)
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

    def vector_size(self):
        return 0


class SparseOnly(RAGStrategy):

    def __init__(self, strategy='bm25', use_gpu=False):
        if strategy == "bm25":
            self.model = BM25TextEmbedding()
        elif strategy == "bm42":
            from tiny_chat.database.embeddings.bm42_embedding import BM42TextEmbedding
            self.model = BM42TextEmbedding(device='cuda' if use_gpu else 'cpu')
        elif strategy == "hotchpotch/japanese-splade-v2":
            from tiny_chat.database.embeddings.splade_embedding import SpladeEmbedding
            self.model = SpladeEmbedding(
                model_name="hotchpotch/japanese-splade-v2", device='cuda' if use_gpu else 'cpu')
        else:
            raise ValueError('unknown Strategy: ' + strategy)
        self.strategy = strategy
        self.sparse_vector_field_name = "sparse"


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
            self.size = self.model.dimension
        elif strategy == "cl-nagoya/ruri-small-v2":
            from tiny_chat.database.embeddings.stransformer_embedding import SentenceTransformerEmbedding
            self.model = SentenceTransformerEmbedding(
                model_name=strategy, device='cuda' if use_gpu else 'cpu')
            self.size = self.model.dimension
        elif strategy == "intfloat/multilingual-e5-large":
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=strategy)
            self.size = 1024
        else:
            from fastembed import TextEmbedding
            self.model = TextEmbedding(model_name=strategy)
            self.size = 512

    def create_vector_config(self):
        return {
            self.dense_vector_field_name: models.VectorParams(
                size=self.vector_size(),
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

    def vector_size(self):
        return self.size


class SpaceRRF(RAGStrategy):
    def __init__(self, strategy='bm25_splide', use_gpu=False):
        if strategy == 'bm25_splide':
            self.sparse_vector_field_names = ["bm25", "splide"]
            from tiny_chat.database.embeddings.splade_embedding import SpladeEmbedding
            self.models = [BM25TextEmbedding(), SpladeEmbedding(
                model_name="hotchpotch/japanese-splade-v2", device='cuda' if use_gpu else 'cpu')]
        else:
            ValueError("unknown strategy: " + strategy)
        self.strategy = strategy

    def create_vector_config(self):
        return {}

    def _get_embeddings(self, text):
        return [list(model.embed(text))[0] for model in self.models]

    def create_sparse_vectors_config(self):
        return {name: models.SparseVectorParams(modifier=models.Modifier.IDF)
                for name in self.sparse_vector_field_names}

    def vector(self, text):
        embeddings = self._get_embeddings(text)
        return {name: embeddings[i].as_object()
                for i, name in enumerate(self.sparse_vector_field_names)}

    def prefetch(self, text, top_k):
        embeddings = self._get_embeddings(text)
        return [models.Prefetch(query=embeddings[i].as_object(), using=name, limit=top_k)
                for i, name in enumerate(self.sparse_vector_field_names)]

    def query(self, text=None):
        return models.FusionQuery(fusion=models.Fusion.RRF)


class SpaceDenseRRF(RAGStrategy):

    def __init__(self, strategy='bm25_static', use_gpu=False):
        if strategy == 'bm25_static':
            self.sparse_vector_field_name = "sparse"
            self.dense_vector_field_name = "dense"
            self.bm25_model = BM25TextEmbedding()
            self.emb_model = StaticEmbedding(device='cuda' if use_gpu else 'cpu')
            self.size = self.emb_model.dimension

        elif strategy == 'bm25_sbert':
            self.sparse_vector_field_name = "sparse"
            self.dense_vector_field_name = "dense"
            self.bm25_model = BM25TextEmbedding()
            from tiny_chat.database.embeddings.stransformer_embedding import SentenceTransformerEmbedding
            self.emb_model = SentenceTransformerEmbedding(device='cuda' if use_gpu else 'cpu')
            self.size = self.emb_model.dimension

        else:
            ValueError("unknown strategy: " + strategy)
        self.strategy = strategy

    def create_vector_config(self):
        return {
            self.dense_vector_field_name: models.VectorParams(
                size=self.vector_size(),
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

    def vector_size(self):
        return self.size
