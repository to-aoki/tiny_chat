from typing import Any, Iterable
import torch
from sentence_transformers import CrossEncoder


class SentenceTransformerCrossEncoder:

    def __init__(
        self,
        model_name: str = "cl-nagoya/ruri-v3-reranker-310m",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        **kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(
            model_name,
            trust_remote_code=True,
            # backend="onnx",
            # model_kwargs={"file_name": "onnx/model_int8.onnx"},
            device=device, **kwargs
        )
        if device.startswith('cuda'):
            self.model.half()

    def rank(
        self,
        query: str,
        documents: Iterable[str],
        **kwargs: Any,
    ) -> Iterable[float]:
        return self.model.rank(query=query, documents=documents)
