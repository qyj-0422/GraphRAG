import asyncio
from typing import Any, List
from Core.Graph.BaseGraph import BaseGraph
from Core.Common.Logger import logger
from Core.Schema.ChunkSchema import TextChunk
from Core.Storage.NetworkXStorage import NetworkXStorage


class ERGraphExisted(BaseGraph):
    """
    A graph loader class for pre-built graph data, skipping LLM extraction process.
    This class directly loads existing graph data from storage instead of building from scratch.
    """

    def __init__(self, config, llm=None, encoder=None):
        """
        Initialize the graph loader with configuration.
        
        Args:
            config: Configuration object containing working directory info
            llm: Optional LLM instance (not used in this implementation)
            encoder: Optional encoder instance (not used in this implementation)
        """
        super().__init__(config, llm, encoder)
        self._graph = NetworkXStorage()
        # Directly load graph during initialization
        asyncio.get_event_loop().run_until_complete(self._load_graph(force=True))

    async def _build_graph(self, chunks: List[Any]):
        """
        Skip graph building process for pre-existing graph.
        
        Args:
            chunks: Input chunks (ignored in this implementation)
        """
        logger.info("Using pre-built graph, skip building process")
        return

    async def _extract_entity_relationship(self, chunk_key_pair: tuple[str, TextChunk]):
        """
        Entity extraction is not supported for pre-built graphs.
        
        Raises:
            NotImplementedError: Always raised since this operation is not supported
        """
        raise NotImplementedError("ERGraphExisted does not support entity extraction as it loads pre-built graphs")

    async def build_graph(self, chunks, force: bool = False):
        """
        Override build_graph to skip building and only load existing graph.
        
        Args:
            chunks: Input chunks (ignored)
            force: Whether to force reload the graph
        """
        logger.info("Loading existing graph from storage")
        await self._load_graph(force)
        logger.info("✅ Finished loading pre-built graph")
