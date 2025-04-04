# pip install yasem
import numpy as np
import torch
import mmh3
from yasem import SpladeEmbedder
from typing import Optional, Union, Iterable, Any
from fastembed import SparseEmbedding


class SpladeEmbedding:

    def __init__(
        self,
        model_name: str = "hotchpotch/japanese-splade-v2",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):

        self.model_name = model_name
        self.device = device
        self.model = SpladeEmbedder(
            model_name,
            device=device,
            max_seq_length=512,
            use_fp16=True,
            **kwargs
        )
        self.dimension = 512

    def embed(
        self,
        documents: Union[str, Iterable[str]],
        batch_size: int = 256,
        parallel: Optional[int] = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:

        if isinstance(documents, str):
            documents = [documents]
        for e in self.model.encode(documents):
            token_values = self.model.get_token_values(e)
            token_ids = [abs(mmh3.hash(token)) for token in list(token_values.keys())]
            indices = np.array(token_ids, dtype=np.int32)
            values = np.array(list(token_values.values()), dtype=np.float32)
            embedding = SparseEmbedding(indices=indices, values=values)
            yield embedding

    def query_embed(
        self, query: Union[str, Iterable[str]], **kwargs: Any
    ) -> Iterable[SparseEmbedding]:
        return self.embed(query)
