"""
Here is the vector index for our GraphRAG
@Author: Yingli
@Reference: 1. https://github.com/geekan/MetaGPT/blob/main/metagpt/rag/engines/simple.py
            2. https://github.com/alibaba/app-controller/blob/main/Core/Index/BaseIndex.py
@Building index:
    1. load index from storage
    2. load index from documents
    3. load index from nodes
"""

from Core.Common.Logger import logger
import os
from typing import Any, Optional, Union

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.readers.base import BaseReader
from llama_index.core import StorageContext, load_index_from_storage, VectorStoreIndex, Settings
from Core.Common.EmbConfig import EmbeddingConfig
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    Document,
    NodeWithScore,
    QueryType,
)

# from C.config2 import config
from Core.Index import (
    get_rag_embedding,
    get_index
)    
from metagpt.rag.interface import NoEmbedding
from metagpt.rag.retrievers.base import ModifiableRAGRetriever, PersistableRAGRetriever
from metagpt.rag.retrievers.hybrid_retriever import SimpleHybridRetriever
# from metagpt.rag.schema import (
#     BaseIndexConfig,
#     BaseRetrieverConfig,
#     BM25RetrieverConfig
# )


class VectorIndex():
    """VectorIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    def __init__(
        self, config
    ) -> None:
        self._index = None
        self.config = config
  
    def build_index(self):


    

        self._index = get_index(self.config)
            # return self.index
        self._update_index(["world", "hello", "your father"])
        self._storage_index()

   
  
    def exist_index(self):
        
        return os.path.exists(self.config.persist_path)

    async def retrieval(self, query, top_k = None):
        if not top_k:
            top_k = self._get_retrieve_top_k()
        retriever: BaseRetriever = self._index.as_retriever(similarity_top_k = top_k)
        nodes = await retriever.aretrieve(query)
        return nodes

    def _update_index(self, list_data: list[Any],type = "document"):
        if type == "document":
            documents = [Document(text=t) for t in list_data]

            self._update_index_from_documents(documents)
        elif type == "node":
            self._update_index_from_nodes(list_data)
        pass

    def _get_retrieve_top_k(self):
        return self.retriever_configs.top_k

    def _storage_index(self):
        self._index.storage_context.persist(persist_dir=self.config.persist_path)



    def _update_index_from_documents(self, docs: list[Any]):
        refreshed_docs = self._index.refresh_ref_docs(docs)

        # the number of docs that are refreshed. if True in refreshed_docs, it means the doc is refreshed.
        logger.info("refresh index size is {}".format(len([True for doc in refreshed_docs if doc])))

    def _update_index_from_nodes(self):
        pass

