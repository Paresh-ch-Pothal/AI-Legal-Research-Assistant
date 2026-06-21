from typing import List

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings


class InLegalBERTEmbeddings(Embeddings):
    """
    Legal-domain embeddings using InLegalBERT
    (fine-tuned on Indian court judgments).
    """

    def __init__(
        self,
        model_name: str = "law-ai/InLegalBERT",
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embedder.embed_query(text)