"""RAG schemas."""
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, List, Literal, Optional, Union

from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.schema import TextNode
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from Core.Config2 import config
from metagpt.configs.embedding_config import EmbeddingType
from metagpt.logs import logger


class BaseRetrieverConfig(BaseModel):
    """Common config for retrievers.

    If add new subconfig, it is necessary to add the corresponding instance implementation in rag.factories.retriever.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    similarity_top_k: int = Field(default=5, description="Number of top-k similar results to return during retrieval.")


class IndexRetrieverConfig(BaseRetrieverConfig):
    """Config for Index-basd retrievers."""

    index: BaseIndex = Field(default=None, description="Index for retriver.")


class FAISSRetrieverConfig(IndexRetrieverConfig):
    """Config for FAISS-based retrievers."""

    dimensions: int = Field(default=0, description="Dimensionality of the vectors for FAISS index construction.")

    _embedding_type_to_dimensions: ClassVar[dict[EmbeddingType, int]] = {
        EmbeddingType.GEMINI: 768,
        EmbeddingType.OLLAMA: 4096,
    }

    @model_validator(mode="after")
    def check_dimensions(self):
        if self.dimensions == 0:
            self.dimensions = config.embedding.dimensions or self._embedding_type_to_dimensions.get(
                config.embedding.api_type, 1536
            )
            if not config.embedding.dimensions and config.embedding.api_type not in self._embedding_type_to_dimensions:
                logger.warning(
                    f"You didn't set dimensions in config when using {config.embedding.api_type}, default to 1536"
                )

        return self


class BM25RetrieverConfig(IndexRetrieverConfig):
    """Config for BM25-based retrievers."""

    _no_embedding: bool = PrivateAttr(default=True)


class MilvusRetrieverConfig(IndexRetrieverConfig):
    """Config for Milvus-based retrievers."""

    uri: str = Field(default="./milvus_local.db", description="The directory to save data.")
    collection_name: str = Field(default="metagpt", description="The name of the collection.")
    token: str = Field(default=None, description="The token for Milvus")
    dimensions: int = Field(default=0, description="Dimensionality of the vectors for Milvus index construction.")

    _embedding_type_to_dimensions: ClassVar[dict[EmbeddingType, int]] = {
        EmbeddingType.GEMINI: 768,
        EmbeddingType.OLLAMA: 4096,
    }

    @model_validator(mode="after")
    def check_dimensions(self):
        if self.dimensions == 0:
            self.dimensions = config.embedding.dimensions or self._embedding_type_to_dimensions.get(
                config.embedding.api_type, 1536
            )
            if not config.embedding.dimensions and config.embedding.api_type not in self._embedding_type_to_dimensions:
                logger.warning(
                    f"You didn't set dimensions in config when using {config.embedding.api_type}, default to 1536"
                )

        return self





class BaseIndexConfig(BaseModel):
    """Common config for index.

    If add new subconfig, it is necessary to add the corresponding instance implementation in rag.factories.index.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    persist_path: Union[str, Path] = Field(description="The directory of saved data.")


class VectorIndexConfig(BaseIndexConfig):
    """Config for vector-based index."""

    embed_model: BaseEmbedding = Field(default=None, description="Embed model.")


class FAISSIndexConfig(VectorIndexConfig):
    """Config for faiss-based index."""




class MilvusIndexConfig(VectorIndexConfig):
    """Config for milvus-based index."""

    collection_name: str = Field(default="metagpt", description="The name of the collection.")
    uri: str = Field(default="./milvus_local.db", description="The uri of the index.")
    token: Optional[str] = Field(default=None, description="The token of the index.")
  


class BM25IndexConfig(BaseIndexConfig):
    """Config for bm25-based index."""

    _no_embedding: bool = PrivateAttr(default=True)





class ParseResultType(str, Enum):
    """The result type for the parser."""

    TXT = "text"
    MD = "markdown"
    JSON = "json"



