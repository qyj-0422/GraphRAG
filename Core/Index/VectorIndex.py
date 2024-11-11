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
from typing import Any
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import (
    BaseNode,
    Document
)
from Core.Index import get_index
from Core.Common.Utils import mdhash_id

class VectorIndex():
    """VectorIndex is designed to be simple and straightforward.

    It is a lightweight and easy-to-use vector database for ANN search.
    """

    def __init__(
        self, config
    ) -> None:
        self._index = None
        self.config = config
        self._index = get_index(self.config)


    async def upsert(self, list_data: list[Any]):
        if len(list_data) == 0:
            logger.warning("No data needs to insert into the vector database")
            return 
        
        if isinstance(list_data[0], Document):
            await self._update_index_from_documents(list_data)
        elif isinstance(list_data[0], BaseNode):
            self._update_index_from_nodes(list_data)
        elif isinstance(list_data[0], str):
            documents = [Document(text = t, key = mdhash_id(t)) for t in list_data]
            await self._update_index_from_documents(documents)
    
    def exist_index(self):
        
        return os.path.exists(self.config.persist_path)

    async def retrieval(self, query, top_k = None):
        if not top_k:
            top_k = self._get_retrieve_top_k()
        retriever: BaseRetriever = self._index.as_retriever(similarity_top_k = top_k)
        nodes = await retriever.aretrieve(query)
        return nodes


    def _get_retrieve_top_k(self):
        return self.retriever_configs.top_k

    def _storage_index(self):
        self._index.storage_context.persist(persist_dir=self.config.persist_path)



    async def _update_index_from_documents(self, docs: list[Any]):
        refreshed_docs = self._index.refresh_ref_docs(docs)

        # the number of docs that are refreshed. if True in refreshed_docs, it means the doc is refreshed.
        logger.info("refresh index size is {}".format(len([True for doc in refreshed_docs if doc])))

    def _update_index_from_nodes(self):
        pass

