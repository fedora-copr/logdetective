"""Embedding service for vector search using fastembed."""
from fastembed import TextEmbedding


class EmbeddingService:
    """Wraps fastembed TextEmbedding for generating vector embeddings."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self._model = TextEmbedding(model_name=model_name)
        self._model_name = model_name

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a vector."""
        results = list(self._model.embed([text]))
        return results[0].tolist()
