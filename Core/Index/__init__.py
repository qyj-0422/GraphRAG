"""RAG factories"""

from Core.Index.EmbeddingFactory import get_rag_embedding
from Core.Index.IndexFactory import get_index


__all__ = ["get_rag_embedding", "get_index"]